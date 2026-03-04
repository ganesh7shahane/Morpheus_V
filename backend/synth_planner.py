"""
Retrosynthetic Planning micro-service (FastAPI).

Runs in the *synplanner* conda environment because the SynPlanner library
and CGRtools are only installed there.

    $SYNPLANNER_PYTHON -m uvicorn synth_planner:app --reload --port 8001
    # e.g. ~/miniconda3/envs/synplanner/bin/python -m uvicorn synth_planner:app --reload --port 8001

Endpoints
---------
POST /plan-synthesis   – run retrosynthetic planning for a SMILES
GET  /health           – liveness check
"""

import gzip
import json
import os
import pickle
import shutil
import sys
import threading
import traceback
from itertools import count, islice
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# SynPlanner imports
# ---------------------------------------------------------------------------
SYNPLANNER_AVAILABLE = False
try:
    from synplan.utils.loading import (
        download_all_data,
        load_building_blocks,
        load_reaction_rules,
        load_policy_function,
        load_evaluation_function,
    )
    from synplan.utils.config import TreeConfig, RolloutEvaluationConfig
    from synplan.chem.utils import mol_from_smiles, safe_canonicalization
    from synplan.mcts.tree import Tree
    from synplan.utils.visualisation import get_route_svg

    SYNPLANNER_AVAILABLE = True
except ImportError:
    pass

CGRTOOLS_AVAILABLE = False
try:
    from CGRtools import smiles as read_cgr_smiles
    from CGRtools.containers.molecule import MoleculeContainer

    CGRTOOLS_AVAILABLE = True
except ImportError:
    pass

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Global cached state
# ---------------------------------------------------------------------------
_data_folder: Optional[Path] = None
_building_blocks = None
_building_blocks_id_map: Dict[str, str] = {}
_reaction_rules = None
_reaction_rules_path: Optional[Path] = None
_policy_function = None
_evaluation_function = None
_tree_config = None
_initializing = False
_init_error: Optional[str] = None

# ---------------------------------------------------------------------------
# Paths – resolved from env vars first, then relative to this repo.
# Set SYNPLAN_DATA_DIR to wherever you store the 2 GB model weights.
# Set BB_SDF_GZ      to the building-blocks SDF.gz if it lives elsewhere.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent  # Morpheus_V/

SYNPLAN_DATA_DIR = Path(
    os.environ.get("SYNPLAN_DATA_DIR",
                   str(_REPO_ROOT / "synplan_data"))
)
BB_SDF_GZ = Path(
    os.environ.get("BB_SDF_GZ",
                   str(_REPO_ROOT / "data" / "building_blocks_em_sa_ln_with_ids.sdf.gz"))
)

REQUIRED_SYNPLANNER_FILES = [
    "uspto/weights/ranking_policy_network.ckpt",
    "uspto/weights/filtering_policy_network.ckpt",
    "uspto/uspto_reaction_rules.pickle",
    "building_blocks/building_blocks_em_sa_ln.smi",
]

# ---------------------------------------------------------------------------
# Helper functions (ported from morpheus/synplanner.py)
# ---------------------------------------------------------------------------


def _check_data_complete(data_folder: Path) -> bool:
    for rel in REQUIRED_SYNPLANNER_FILES:
        if not (data_folder / rel).exists():
            return False
    return True


def _ensure_data(data_folder: Optional[Path] = None) -> Path:
    global _data_folder
    if data_folder is None:
        data_folder = SYNPLAN_DATA_DIR
    data_folder = Path(data_folder).resolve()
    if not data_folder.exists() or not _check_data_complete(data_folder):
        print(f"Downloading SynPlanner data to {data_folder} …")
        download_all_data(save_to=data_folder)
    _data_folder = data_folder
    return data_folder


def _load_bb_with_ids(sdf_path: Path):
    """Load building blocks from SDF and return (frozenset, smiles→id dict)."""
    global _building_blocks_id_map
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to load building blocks from SDF")

    cache_path = sdf_path.with_suffix(".pickle")
    if cache_path.exists() and cache_path.stat().st_mtime >= sdf_path.stat().st_mtime:
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            _building_blocks_id_map = cached["smiles_to_id"]
            print(f"Loaded {len(cached['building_blocks'])} BBs from cache")
            return cached["building_blocks"], cached["smiles_to_id"]
        except Exception:
            pass

    print("Parsing building blocks SDF (first run) …")
    bb_set: set = set()
    s2id: Dict[str, str] = {}
    suppl = Chem.SDMolSupplier(str(sdf_path))
    for idx, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            mol_id = mol.GetProp("ID") if mol.HasProp("ID") else mol.GetProp("_Name")
            rdkit_smi = Chem.MolToSmiles(mol)
            if CGRTOOLS_AVAILABLE:
                try:
                    cgr_mol = read_cgr_smiles(rdkit_smi)
                    cgr_mol = safe_canonicalization(cgr_mol)
                    canonical = str(cgr_mol)
                except Exception:
                    canonical = rdkit_smi
            else:
                canonical = rdkit_smi
            bb_set.add(canonical)
            s2id[canonical] = mol_id
            if rdkit_smi != canonical:
                s2id[rdkit_smi] = mol_id
            if (idx + 1) % 50000 == 0:
                print(f"  {idx + 1}/{len(suppl)} …")
        except Exception:
            continue
    _building_blocks_id_map = s2id
    frozen = frozenset(bb_set)
    print(f"Loaded {len(bb_set)} BBs with {len(s2id)} ID mappings")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"building_blocks": frozen, "smiles_to_id": s2id}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return frozen, s2id


def _get_bb_id(smiles: str) -> Optional[str]:
    if smiles in _building_blocks_id_map:
        return _building_blocks_id_map[smiles]
    if CGRTOOLS_AVAILABLE:
        try:
            cgr = read_cgr_smiles(smiles)
            cgr = safe_canonicalization(cgr)
            c = str(cgr)
            if c in _building_blocks_id_map:
                return _building_blocks_id_map[c]
        except Exception:
            pass
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                c = Chem.MolToSmiles(mol)
                if c in _building_blocks_id_map:
                    return _building_blocks_id_map[c]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# SVG with BB-IDs (ported from morpheus/synplanner.py)
# ---------------------------------------------------------------------------

def _render_svg_with_labels(pred, columns, box_colors, smiles_to_id=None):
    if not CGRTOOLS_AVAILABLE:
        return None
    x_shift = 0.0
    c_max_x = 0.0
    c_max_y = 0.0
    render = []
    cx = count()
    cy = count()
    arrow_points = {}
    label_positions = []
    intermediate_positions = []
    intermediate_counter = count(1)

    for ms in columns:
        heights = []
        for m in ms:
            m.clean2d()
            min_x = min(x for x, y in m._plane.values()) - x_shift
            min_y = min(y for x, y in m._plane.values())
            m._plane = {n: (x - min_x, y - min_y) for n, (x, y) in m._plane.items()}
            max_x = max(x for x, y in m._plane.values())
            c_max_x = max(c_max_x, max_x)
            arrow_points[next(cx)] = [x_shift, max_x]
            heights.append(max(y for x, y in m._plane.values()))
        x_shift = c_max_x + 5.0
        y_shift = sum(heights) + 3.0 * (len(heights) - 1)
        c_max_y = max(c_max_y, y_shift)
        y_shift /= 2.0
        for m, h in zip(ms, heights):
            m._plane = {n: (x, y - y_shift) for n, (x, y) in m._plane.items()}
            max_x = max(x for x, y in m._plane.values()) + 0.9
            min_x = min(x for x, y in m._plane.values()) - 0.6
            max_y = -(max(y for x, y in m._plane.values()) + 0.45)
            min_y = -(min(y for x, y in m._plane.values()) - 0.45)
            x_delta = abs(max_x - min_x)
            y_delta = abs(max_y - min_y)
            box = (
                f'<rect x="{min_x}" y="{max_y}" rx="{y_delta*0.1}" ry="{y_delta*0.1}" '
                f'width="{x_delta}" height="{y_delta}" stroke="black" stroke-width=".0025" '
                f'fill="{box_colors[m.meta["status"]]}" fill-opacity="0.30"/>'
            )
            if m.meta.get("status") == "instock" and smiles_to_id:
                smi = str(m)
                bb_id = smiles_to_id.get(smi) or _get_bb_id(smi)
                if bb_id:
                    lx = (min_x + max_x) / 2
                    ly = min_y + 0.50
                    label_positions.append((lx, ly, bb_id))
            elif m.meta.get("status") == "mulecule":
                lx = (min_x + max_x) / 2
                ly = min_y + 0.50
                intermediate_positions.append((lx, ly, f"Int-{next(intermediate_counter)}"))
            arrow_points[next(cy)].append(y_shift - h / 2.0)
            y_shift -= h + 3.0
            depicted = list(m.depict(embedding=True))[:3]
            depicted.append(box)
            render.append(depicted)

    graph: dict = {}
    for s, p in pred:
        graph.setdefault(s, []).append(p)
    for s, ps in graph.items():
        mid_x = float("-inf")
        for p in ps:
            s_min_x, s_max, s_y = arrow_points[s][:3]
            p_min_x, p_max, p_y = arrow_points[p][:3]
            p_max += 1
            mid = p_max + (s_min_x - p_max) / 3
            mid_x = max(mid_x, mid)
        for p in ps:
            arrow_points[p].append(mid_x)

    config = MoleculeContainer._render_config
    font_size = config["font_size"]
    font125 = 1.25 * font_size
    width = c_max_x + 4.0 * font_size
    height = c_max_y + 3.5 * font_size
    box_y = height / 2.0

    svg = [
        f'<svg width="{0.6*width:.2f}cm" height="{0.8*height:.2f}cm" '
        f'viewBox="{-font125:.2f} {-box_y:.2f} {width:.2f} '
        f'{height:.2f}" xmlns="http://www.w3.org/2000/svg" version="1.1">',
        '  <defs>\n    <marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="0" refY="3" orient="auto">\n      <path d="M0,0 L0,6 L9,3"/>\n    </marker>\n  </defs>',
    ]
    for s, p in pred:
        s_min_x, s_max, s_y = arrow_points[s][:3]
        p_min_x, p_max, p_y = arrow_points[p][:3]
        p_max += 1
        mid_x = arrow_points[p][-1]
        arrow = (
            f'  <polyline points="{p_max:.2f} {p_y:.2f}, {mid_x:.2f} {p_y:.2f}, '
            f'{mid_x:.2f} {s_y:.2f}, {s_min_x - 1.:.2f} {s_y:.2f}" '
            f'fill="none" stroke="black" stroke-width=".04" marker-end="url(#arrow)"/>'
        )
        if p_y != s_y:
            arrow += f'  <circle cx="{mid_x}" cy="{p_y}" r="0.1"/>'
        svg.append(arrow)
    for atoms, bonds, masks, box in render:
        mol_svg = MoleculeContainer._graph_svg(atoms, bonds, masks, -font125, -box_y, width, height)
        mol_svg.insert(1, box)
        svg.extend(mol_svg)
    for lx, ly, bb_id in label_positions:
        svg.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" '
            f'font-size="0.60" font-family="sans-serif" fill="#006400" font-weight="bold">{bb_id}</text>'
        )
    for lx, ly, lbl in intermediate_positions:
        svg.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" '
            f'font-size="0.60" font-family="sans-serif" fill="#e65100" font-weight="bold">{lbl}</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


def _get_route_svg_with_ids(tree, node_id, smiles_to_id=None):
    if not CGRTOOLS_AVAILABLE:
        return get_route_svg(tree, node_id)
    if node_id not in tree.winning_nodes:
        return None
    nodes = tree.route_to_node(node_id)
    if smiles_to_id is None:
        smiles_to_id = _building_blocks_id_map
    for n in nodes:
        for pr in n.new_precursors:
            pr.molecule.meta["status"] = (
                "instock" if pr.is_building_block(tree.building_blocks) else "mulecule"
            )
    nodes[0].curr_precursor.molecule.meta["status"] = "target"
    box_colors = {"target": "#98EEFF", "mulecule": "#F0AB90", "instock": "#9BFAB3"}
    columns = [
        [nodes[0].curr_precursor.molecule],
        [x.molecule for x in nodes[1].new_precursors],
    ]
    pred = {x: 0 for x in range(1, len(columns[1]) + 1)}
    cx_list = [
        n for n, x in enumerate(nodes[1].new_precursors, 1)
        if not x.is_building_block(tree.building_blocks)
    ]
    size = len(cx_list)
    nodes_iter = iter(nodes[2:])
    cy_counter = count(len(columns[1]) + 1)
    while size:
        layer = []
        for s_node in islice(nodes_iter, size):
            n_idx = cx_list.pop(0)
            for x in s_node.new_precursors:
                layer.append(x)
                m_idx = next(cy_counter)
                if not x.is_building_block(tree.building_blocks):
                    cx_list.append(m_idx)
                pred[m_idx] = n_idx
        size = len(cx_list)
        columns.append([x.molecule for x in layer])
    columns = [col[::-1] for col in columns[::-1]]
    pred_tuples = tuple(
        (abs(src - len(pred)), abs(tgt - len(pred))) for tgt, src in pred.items()
    )
    return _render_svg_with_labels(pred_tuples, columns, box_colors, smiles_to_id)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize(bb_sdf_gz: Optional[Path] = None) -> bool:
    global _building_blocks, _reaction_rules, _reaction_rules_path
    global _policy_function, _evaluation_function, _tree_config, _data_folder

    # Reset all globals so a failed init never leaves a partially-initialized state
    # (e.g. _building_blocks set but _tree_config still None)
    _building_blocks = None
    _reaction_rules = None
    _reaction_rules_path = None
    _policy_function = None
    _evaluation_function = None
    _tree_config = None

    if not SYNPLANNER_AVAILABLE:
        print("SynPlanner not installed")
        return False
    try:
        data_folder = _ensure_data()

        # Building blocks
        bb_path = bb_sdf_gz or BB_SDF_GZ
        bb_path = Path(bb_path)
        if str(bb_path).endswith(".sdf.gz"):
            decomp = Path(str(bb_path)[:-3])
            if not decomp.exists():
                print(f"Decompressing {bb_path.name} …")
                with gzip.open(bb_path, "rb") as fi, open(decomp, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            bb_path = decomp
        if bb_path.exists():
            _building_blocks, _ = _load_bb_with_ids(bb_path)
        else:
            fallback = data_folder / "building_blocks" / "building_blocks_em_sa_ln.smi"
            _building_blocks = load_building_blocks(fallback, standardize=False)

        # Reaction rules
        _reaction_rules_path = data_folder / "uspto" / "uspto_reaction_rules.pickle"
        _reaction_rules = load_reaction_rules(_reaction_rules_path)
        print(f"Loaded {len(_reaction_rules)} reaction rules")

        # Policy
        policy_path = data_folder / "uspto" / "weights" / "ranking_policy_network.ckpt"
        _policy_function = load_policy_function(weights_path=policy_path)
        print("Loaded policy network")

        # Tree config (defaults; per-request overrides possible)
        _tree_config = TreeConfig(
            search_strategy="expansion_first",
            max_iterations=300,
            max_time=120,
            max_depth=9,
            min_mol_size=1,
            init_node_value=0.5,
            ucb_type="uct",
            c_ucb=0.1,
        )

        # Evaluation function
        eval_cfg = RolloutEvaluationConfig(
            policy_network=_policy_function,
            reaction_rules=_reaction_rules,
            building_blocks=_building_blocks,
            min_mol_size=_tree_config.min_mol_size,
            max_depth=_tree_config.max_depth,
        )
        _evaluation_function = load_evaluation_function(eval_cfg)
        print("Initialized evaluation function")
        return True
    except Exception as e:
        traceback.print_exc()
        return False


def plan_synthesis(
    smiles: str,
    max_routes: int = 5,
    return_svg: bool = True,
    max_depth: Optional[int] = None,
    max_iterations: Optional[int] = None,
    min_mol_size: Optional[int] = None,
) -> Dict[str, Any]:
    global _building_blocks, _tree_config

    if not SYNPLANNER_AVAILABLE:
        return {"success": False, "solved": False, "routes": [], "error": "SynPlanner not installed"}

    if _building_blocks is None or _tree_config is None:
        if not initialize():
            return {"success": False, "solved": False, "routes": [], "error": "Failed to init SynPlanner"}

    try:
        target = mol_from_smiles(smiles, clean2d=True, standardize=True, clean_stereo=True)
        if target is None:
            return {"success": False, "solved": False, "routes": [], "error": f"Cannot parse SMILES: {smiles}"}

        tc = _tree_config
        if any(v is not None for v in (max_depth, max_iterations, min_mol_size)):
            tc = TreeConfig(
                search_strategy=tc.search_strategy,
                max_iterations=max_iterations or tc.max_iterations,
                max_time=tc.max_time,
                max_depth=max_depth or tc.max_depth,
                min_mol_size=min_mol_size if min_mol_size is not None else tc.min_mol_size,
                init_node_value=tc.init_node_value,
                ucb_type=tc.ucb_type,
                c_ucb=tc.c_ucb,
            )

        # Rebuild evaluation function if min_mol_size or max_depth changed
        eval_fn = _evaluation_function
        if (max_depth and max_depth != _tree_config.max_depth) or \
           (min_mol_size is not None and min_mol_size != _tree_config.min_mol_size):
            eval_cfg = RolloutEvaluationConfig(
                policy_network=_policy_function,
                reaction_rules=_reaction_rules,
                building_blocks=_building_blocks,
                min_mol_size=tc.min_mol_size,
                max_depth=tc.max_depth,
            )
            eval_fn = load_evaluation_function(eval_cfg)

        tree = Tree(
            target=target,
            config=tc,
            reaction_rules=_reaction_rules,
            building_blocks=_building_blocks,
            expansion_function=_policy_function,
            evaluation_function=eval_fn,
        )

        solved = False
        for s, nid in tree:
            if s:
                solved = True

        routes: List[Dict[str, Any]] = []
        if solved and hasattr(tree, "winning_nodes"):
            all_info = []
            for nid in tree.winning_nodes:
                try:
                    nodes = tree.route_to_node(nid)
                    depth = int(len(nodes) - 1)
                except Exception:
                    depth = 999
                all_info.append({"node_id": nid, "depth": depth, "score": float(tree.route_score(nid))})
            all_info.sort(key=lambda x: (x["depth"], -x["score"]))

            for ri, rd in enumerate(all_info):
                if ri >= max_routes:
                    break
                nid = rd["node_id"]
                info: Dict[str, Any] = {"node_id": int(nid), "score": float(rd["score"]), "num_steps": int(rd["depth"])}
                try:
                    nodes = tree.route_to_node(nid)
                    bbs: list = []
                    intermediates: list = []
                    seen: set = set()
                    target_smi = str(nodes[0].curr_precursor.molecule) if nodes else None
                    for node in nodes:
                        for pr in node.new_precursors:
                            ms = str(pr.molecule)
                            if ms in seen:
                                continue
                            seen.add(ms)
                            if pr.is_building_block(tree.building_blocks):
                                bbs.append({"smiles": ms, "id": _get_bb_id(ms)})
                            elif ms != target_smi:
                                intermediates.append({"smiles": ms})
                    info["building_blocks"] = bbs
                    info["intermediates"] = intermediates
                except Exception:
                    info["building_blocks"] = []
                    info["intermediates"] = []

                if return_svg:
                    try:
                        svg = _get_route_svg_with_ids(tree, nid, _building_blocks_id_map)
                        if svg is None:
                            svg = get_route_svg(tree, nid)
                        info["svg"] = svg
                    except Exception:
                        try:
                            info["svg"] = get_route_svg(tree, nid)
                        except Exception as e2:
                            info["svg"] = None
                            info["svg_error"] = str(e2)
                routes.append(info)

        _iters = getattr(tree, "curr_iteration", None)
        return {
            "success": True,
            "solved": bool(solved),
            "routes": routes,
            "num_iterations": int(_iters) if _iters is not None else None,
            "target_smiles": smiles,
        }
    except BaseException as e:  # noqa: BLE001 – catch SystemExit, MemoryError, etc.
        traceback.print_exc()
        return {"success": False, "solved": False, "routes": [], "error": str(e)}


# ===========================================================================
# FastAPI application
# ===========================================================================

app = FastAPI(title="Morpheus V – Retrosynthetic Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _json_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: always return JSON so the proxy never sees a raw HTML 500.

    NOTE: Starlette requires the type to be a subclass of Exception, not
    BaseException, so we register only Exception here.  The endpoint-level
    handlers use `except BaseException` to cover SystemExit / MemoryError.
    """
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "solved": False,
            "routes": [],
            "error": f"{type(exc).__name__}: {exc}",
        },
    )


@app.on_event("startup")
def _startup_init():
    """Begin loading building blocks + models in a background thread so the
    server starts accepting connections immediately (health checks, etc.)."""
    def _bg():
        global _initializing, _init_error
        _initializing = True
        _init_error = None
        try:
            ok = initialize()
            if not ok:
                _init_error = "initialize() returned False"
        except BaseException as exc:  # noqa: BLE001
            _init_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            _initializing = False

    threading.Thread(target=_bg, daemon=True).start()


class PlanRequest(BaseModel):
    smiles: str
    max_routes: int = 5
    max_depth: int = 4
    max_iterations: int = 200
    min_mol_size: int = 1


@app.get("/health")
def health():
    return {
        "status": "ok",
        "synplanner_available": SYNPLANNER_AVAILABLE,
        "initialized": _building_blocks is not None,
        "initializing": _initializing,
        "init_error": _init_error,
    }


@app.post("/plan-synthesis")
def api_plan_synthesis(req: PlanRequest):
    try:
        result = plan_synthesis(
            smiles=req.smiles,
            max_routes=req.max_routes,
            return_svg=True,
            max_depth=req.max_depth,
            max_iterations=req.max_iterations,
            min_mol_size=req.min_mol_size,
        )
        # Validate JSON serializability before FastAPI touches the object.
        # Use jsonable_encoder which handles numpy/pydantic types the same
        # way FastAPI does, guaranteeing no serialization surprise downstream.
        try:
            from fastapi.encoders import jsonable_encoder as _enc
            import json as _json
            _json.dumps(_enc(result))
        except Exception as _se:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "solved": False,
                    "routes": [],
                    "error": f"Result contained non-serializable data: {_se}",
                },
            )
        return result
    except BaseException as _exc:  # noqa: BLE001 – catch SystemExit, MemoryError, etc.
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "solved": False,
                "routes": [],
                "error": f"{type(_exc).__name__}: {_exc}",
            },
        )
