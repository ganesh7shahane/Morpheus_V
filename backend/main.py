"""
FastAPI backend for Morpheus V – molecule fragmentation service.
"""

import os, sys, io, base64, json, asyncio, copy, traceback
from pathlib import Path
import requests as _requests
import httpx as _httpx

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdDepictor, Draw, AllChem, rdDistGeom, rdFingerprintGenerator, DataStructs, rdMolDescriptors, rdFMCS, rdDetermineBonds
# Local patched copy of sascorer — uses the modern rdFingerprintGenerator API
# to avoid the RDKit deprecation warning "please use MorganGenerator".
import sascorer as _sascorer

try:
    from espsim import EmbedAlignScore as _EmbedAlignScore
    _ESPSIM_AVAILABLE = True
except ImportError:
    _ESPSIM_AVAILABLE = False

from fragmentation import (
    decompose_molecule_with_wildcards,
    mol_to_base64_png,
    count_heavy_atoms,
)
from search_replace import (
    find_similar_fragments,
    reassemble_from_smiles,
    mol_to_base64_png as sr_mol_to_png,
)

FRAGMENTS_FILE = str(Path(__file__).parent / "data" / "All-In-One.txt.gz")
DATA_DIR = Path(__file__).parent / "data"
ALL_FRAGMENT_LIBRARIES: list[str] = sorted(p.name for p in DATA_DIR.glob("*.txt.gz"))

# ============================================================================
# UNDESIRABLE SMARTS PATTERNS (Structural Alerts)
# ============================================================================
UNDESIRABLE_PATTERNS = [
    # Radioactive isotopes
    ('[18F]', 'Fluorine-18 (radioactive)'),
    ('[11C]', 'Carbon-11 (radioactive)'),
    ('[123I]', 'Iodine-123 (radioactive)'),
    # Peroxides and related
    ('[O]-[O]', 'Peroxide'),
    ('[O]-[O]-[O]', 'Ozonide'),
    ('C(=O)O[O]', 'Peroxycarboxylate'),
    ('C(=O)OO', 'Peroxyacid'),
    # Nitrogen-nitrogen bonds
    ('[n]-[N]', 'Connected Ring Nitrogens'),
    ('[N]-[N]', 'Hydrazine (N-N)'),
    ('[N]=[N]-[N]', 'Azide (N=N-N)'),
    # Disulfide
    ('[S]-[S]', 'Disulfide (S-S)'),
    # N-O bonds
    ('[n]-[O]', 'n-O bond'),
    ('[O]-[N]', 'O-N bond'),
    ('[N]-[O]', 'N-O bond'),
    # Acyl halides
    ('C(=O)Cl', 'Acyl Chloride'),
    ('C(=O)Br', 'Acyl Bromide'),
    ('C(=O)F', 'Acyl Fluoride'),
    # Sulfonyl chloride
    ('[S](=O)(=O)Cl', 'Sulfonyl Chloride'),
    # Phosphorus chlorides
    ('[P]Cl', 'Phosphorus Chloride'),
    ('P(=O)(Cl)(Cl)', 'Phosphoryl Dichloride'),
    ('P(Cl)(Cl)(Cl)', 'Phosphorus Trichloride'),
    # Mixed anhydride / acyl-O-alkyl with adjacent acyl
    ('C(=O)OC(=O)', 'Anhydride'),
    ('C(=O)O[C;!$(C=O)]', 'Acyl-O-alkyl (ester)'),
    # Aldehydes
    ('[CH]=O', 'Aldehyde'),
    ('[CX3H1](=O)[#6]', 'Aldehyde'),
    # Nitro groups
    ('[N+](=O)[O-]', 'Nitro group'),
    ('[NX3](=O)=O', 'Aromatic Nitro'),
    # Nitro adjacent to carbonyl
    ('[N+](=O)[O-]C(=O)', 'Nitro adjacent to carbonyl'),
    ('C(=O)C[N+](=O)[O-]', 'Carbonyl adjacent to nitro'),
    # Isocyanate and Isothiocyanate
    ('N=C=O', 'Isocyanate'),
    ('N=C=S', 'Isothiocyanate'),
    # Thiol
    ('[SH]', 'Thiol'),
    # Cyanohydrin motif (carbon with both OH and CN)
    ('[CH]([OH])(C#N)', 'Cyanohydrin'),
    # Phenol
    ('c[OH]', 'Phenol'),
    # Michael acceptors (alpha,beta-unsaturated carbonyl)
    ('[#6]=[#6]-C(=O)', 'Michael acceptor'),
    # Quinone-like (redox active)
    ('c1cc(=O)cc(=O)c1', 'Quinone (redox active)'),
    ('C1=CC(=O)C=CC1=O', 'Benzoquinone'),
    # Rhodanine-ish (PAINS)
    ('O=C1NC(=O)C=C1', 'Rhodanine-like (PAINS)'),
    ('O=C1NC(=S)SC1', 'Rhodanine'),
    ('O=C1NC(=O)SC1', 'Thiazolidinedione'),
    # Problematic ring systems
    ("O=C1OCC([*:1])=C1[*:3]", 'Butenolide'),
    ("c1(/C=C/c2ccccc2)ccccc1", '1,2-Diphenylethylene (Stilbene)'),
    # Strained N-containing rings
    ('C1CN1', 'Aziridine (strained 3-membered N-ring)'),
    # Nitroso groups
    ('[#6][NX2]=O', 'Nitroso group (C-N=O)'),
    ('[N][N]=O', 'N-Nitroso / Nitrosamine (carcinogenic)'),
    # Polycyclic aromatic hydrocarbons (PAH) – 3 or more fused aromatic rings
    # Anthracene SMARTS covers linear tricyclics and larger (tetracene, pyrene, etc.)
    ('c1ccc2cc3ccccc3cc2c1', 'PAH: linear 3+ fused aromatic rings (anthracene-type)'),
    # Phenanthrene SMARTS covers angular tricyclics and larger (chrysene, pyrene, etc.)
    ('c1ccc2c(c1)ccc1ccccc12', 'PAH: angular 3+ fused aromatic rings (phenanthrene-type)'),
    # Large aliphatic macrocyclic rings (> 8 members) – poor oral bioavailability / permeability
    ('[!a;r9]',  'Large aliphatic ring (9-membered macrocycle)'),
    ('[!a;r10]', 'Large aliphatic ring (10-membered macrocycle)'),
    ('[!a;r11]', 'Large aliphatic ring (11-membered macrocycle)'),
    ('[!a;r12]', 'Large aliphatic ring (12-membered macrocycle)'),
    ('[!a;r13]', 'Large aliphatic ring (13-membered macrocycle)'),
    ('[!a;r14]', 'Large aliphatic ring (14-membered macrocycle)'),
    ('[!a;r15]', 'Large aliphatic ring (15-membered macrocycle)'),
    ('[!a;r16]', 'Large aliphatic ring (16-membered macrocycle)'),
]

app = FastAPI(title="Morpheus V API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _eager_start_synth_planner() -> None:
    """Kick off synth_planner startup in the background as soon as the main
    backend starts, so building blocks are already loaded (or loading) by the
    time the user clicks 'Run Retrosynthetic Planning'."""
    def _bg() -> None:
        try:
            if _start_synth_planner():
                print("[main] synth_planner service started; waiting for init…")
                ok, err = _wait_for_synth_init(timeout=600)
                if ok:
                    print("[main] synth_planner fully initialised and ready.")
                else:
                    print(f"[main] synth_planner init failed: {err}")
            else:
                print("[main] synth_planner could not be started (synplanner env may be missing).")
        except Exception as exc:  # noqa: BLE001
            print(f"[main] synth_planner eager-start error: {exc}")

    threading.Thread(target=_bg, daemon=True, name="synth_planner_eager_start").start()


# ---------- request / response models ----------

class SmilesRequest(BaseModel):
    smiles: str
    max_terminal_atoms: int = 3


class FragmentInfo(BaseModel):
    wildcard_smiles: str
    base_smiles: str
    frag_type: str
    size: int
    hetero_count: int
    image: str  # base64 PNG
    all_fragment_index: int  # index of this fragment in the full (all_fragment_smiles) list


class MoleculeInfo(BaseModel):
    smiles: str
    image: str  # base64 PNG
    molblock_3d: str | None = None  # SDF mol block for 3D viewer
    num_rings: int
    mw: float
    hbd: int
    hba: int
    tpsa: float
    clogp: float
    qed: float
    sascore: float | None = None


class FragmentationResponse(BaseModel):
    molecule: MoleculeInfo
    fragments: list[FragmentInfo]
    all_fragment_smiles: list[str]  # ALL fragments (including small) needed for reassembly
    total_fragments: int
    displayed_fragments: int


class ValidateResponse(BaseModel):
    valid: bool
    canonical: str | None = None
    error: str | None = None


class SearchReplaceRequest(BaseModel):
    query_smiles: str  # the selected fragment (with wildcards)
    all_fragment_smiles: list[str]  # all fragments for the molecule
    selected_index: int  # which fragment to replace
    original_smiles: str  # the original input molecule SMILES
    similarity_threshold: float = 0.3
    top_n: int = 50
    library_names: list[str] = []  # empty = use default ChEMBL file


class GeneratedMolecule(BaseModel):
    smiles: str
    image: str  # base64 PNG
    new_fragment_smiles: str = ""
    frag_similarity: float
    mol_similarity: float | None = None
    shape_sim: float | None = None
    esp_sim: float | None = None
    shape_esp: float | None = None
    mw: float | None = None
    clogp: float | None = None
    qed: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    n_aliphatic_rings: int | None = None
    n_aromatic_rings: int | None = None
    n_rotatable_bonds: int | None = None
    sascore: float | None = None
    mscore: float | None = None


class RejectedMolecule(BaseModel):
    smiles: str
    image: str  # base64 PNG
    frag_similarity: float
    mol_similarity: float | None = None
    shape_sim: float | None = None
    esp_sim: float | None = None
    shape_esp: float | None = None
    mw: float | None = None
    clogp: float | None = None
    qed: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    n_aliphatic_rings: int | None = None
    n_aromatic_rings: int | None = None
    n_rotatable_bonds: int | None = None
    sascore: float | None = None
    mscore: float | None = None
    rejection_reasons: list[str]  # names of matched UNDESIRABLE_PATTERNS


# Pre-compile UNDESIRABLE_PATTERNS for fast screening
_COMPILED_PATTERNS: list[tuple] = []
for _smarts, _name in UNDESIRABLE_PATTERNS:
    _pat = Chem.MolFromSmarts(_smarts)
    if _pat is not None:
        _COMPILED_PATTERNS.append((_pat, _name))


def screen_molecule(mol) -> list[str]:
    """Return list of matched undesirable pattern names (empty = clean)."""
    matched = []
    for pat, name in _COMPILED_PATTERNS:
        try:
            if mol.HasSubstructMatch(pat):
                matched.append(name)
        except Exception:
            pass
    return matched


class SearchReplaceResponse(BaseModel):
    similar_fragments: list[dict]  # [{smiles, similarity, image}]
    generated_molecules: list[GeneratedMolecule]
    total_similar: int


class PatternInfo(BaseModel):
    smarts: str
    name: str
    image: str  # base64 PNG, empty string if render failed


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/undesirable-patterns", response_model=list[PatternInfo])
def get_undesirable_patterns():
    """Return all UNDESIRABLE_PATTERNS with rendered SMARTS images."""
    result = []
    for smarts, name in UNDESIRABLE_PATTERNS:
        pattern_mol = Chem.MolFromSmarts(smarts)
        img_b64 = ""
        if pattern_mol:
            try:
                rdDepictor.Compute2DCoords(pattern_mol)
                img = Draw.MolToImage(pattern_mol, size=(300, 300))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass
        result.append(PatternInfo(smarts=smarts, name=name, image=img_b64))
    return result


@app.get("/fragment-libraries")
def get_fragment_libraries():
    """Return names of all available fragment library files in the data directory."""
    return {"libraries": ALL_FRAGMENT_LIBRARIES}


# ---------- search & replace (SSE stream for progress) ----------

@app.post("/search-replace")
async def search_and_replace(req: SearchReplaceRequest):
    """Stream progress via SSE, then return results in final event."""

    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_q: asyncio.Queue = asyncio.Queue()

        # Resolve which library files to search
        _lib_paths = (
            [str(DATA_DIR / name) for name in req.library_names if (DATA_DIR / name).exists()]
            if req.library_names
            else [FRAGMENTS_FILE]
        ) or [FRAGMENTS_FILE]
        _n_libs = len(_lib_paths)

        def _search_all_libraries():
            _seen: dict[str, tuple[float, int, str]] = {}  # smi -> (sim, na, lib_name)
            for _lib_idx, _lib_path in enumerate(_lib_paths):
                _lib_name = Path(_lib_path).name
                def _cb(p: float, _i=_lib_idx):
                    try:
                        progress_q.put_nowait((_i + p) / _n_libs * 0.5)
                    except:
                        pass
                for _smi, _sim, _na in find_similar_fragments(
                    req.query_smiles,
                    _lib_path,
                    similarity_threshold=req.similarity_threshold,
                    top_n=req.top_n,
                    progress_callback=_cb,
                ):
                    if _smi not in _seen or _sim > _seen[_smi][0]:
                        _seen[_smi] = (_sim, _na, _lib_name)
            return sorted(
                [(_smi, _sim, _na, _lib) for _smi, (_sim, _na, _lib) in _seen.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:req.top_n]

        # Run CPU-heavy search in a thread
        similar_future = loop.run_in_executor(None, _search_all_libraries)

        # Stream progress events while waiting
        while not similar_future.done():
            try:
                p = await asyncio.wait_for(progress_q.get(), timeout=0.3)
                yield f"data: {json.dumps({'type': 'progress', 'value': round(p, 4)})}\n\n"
            except asyncio.TimeoutError:
                pass

        similar = await similar_future
        yield f"data: {json.dumps({'type': 'progress', 'value': 0.5, 'stage': 'generating'})}\n\n"

        # Build similar-fragment images
        sim_frags = []
        for entry in similar:
            smi, sim, na, lib = entry
            fm = Chem.MolFromSmiles(smi)
            img = sr_mol_to_png(fm, size=(500, 500)) if fm else ""
            sim_frags.append({"smiles": smi, "similarity": round(sim, 4), "image": img, "library": lib})

        # ---- Reassemble molecules ----
        ref_mol = Chem.MolFromSmiles(req.original_smiles)
        ref_fp = None
        if ref_mol:
            AllChem.Compute2DCoords(ref_mol)
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            ref_fp = fpgen.GetFingerprint(ref_mol)

        # Match morpheus.py: only screen *generated* molecules if the input
        # molecule itself is already clean (doesn't have those patterns).
        ref_has_undesirable = bool(screen_molecule(ref_mol)) if ref_mol else False

        generated: list[dict] = []
        rejected: list[dict] = []
        seen_smiles: set[str] = set()
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        _n_similar = len(similar)

        for _gen_idx, entry in enumerate(similar):
            smi, sim, _, lib = entry
            new_frags = list(req.all_fragment_smiles)
            new_frags[req.selected_index] = smi
            new_mol = reassemble_from_smiles(new_frags)
            if new_mol is None:
                continue
            new_smi = Chem.MolToSmiles(new_mol, canonical=True)
            if new_smi in seen_smiles:
                continue
            seen_smiles.add(new_smi)

            # Align to reference
            if ref_mol:
                try:
                    AllChem.GenerateDepictionMatching2DStructure(new_mol, ref_mol)
                except:
                    try:
                        AllChem.Compute2DCoords(new_mol)
                    except:
                        pass

            mol_sim = None
            if ref_fp:
                try:
                    mol_sim = round(DataStructs.TanimotoSimilarity(ref_fp, fpgen.GetFingerprint(new_mol)), 4)
                except:
                    pass

            try:
                mw = round(Descriptors.MolWt(new_mol), 2)
            except:
                mw = None
            try:
                clogp = round(Crippen.MolLogP(new_mol), 2)
            except:
                clogp = None
            try:
                qed_val = round(QED.qed(new_mol), 3)
            except:
                qed_val = None
            try:
                tpsa = round(Descriptors.TPSA(new_mol), 2)
            except:
                tpsa = None
            try:
                hbd = Descriptors.NumHDonors(new_mol)
            except:
                hbd = None
            try:
                hba = Descriptors.NumHAcceptors(new_mol)
            except:
                hba = None
            try:
                n_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(new_mol)
            except:
                n_aliphatic_rings = None
            try:
                n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(new_mol)
            except:
                n_aromatic_rings = None
            try:
                n_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(new_mol)
            except:
                n_rotatable_bonds = None
            try:
                sascore = round(_sascorer.calculateScore(new_mol), 2)
            except:
                sascore = None

            mscore = round(mol_sim * qed_val, 3) if mol_sim is not None and qed_val is not None else None

            mol_entry = {
                "smiles": new_smi,
                "image": sr_mol_to_png(new_mol, size=(600, 600)),
                "new_fragment_smiles": smi,
                "frag_library": lib,
                "frag_similarity": round(sim, 4),
                "mol_similarity": mol_sim,
                "shape_sim": None,
                "esp_sim": None,
                "shape_esp": None,
                "mw": mw,
                "clogp": clogp,
                "qed": qed_val,
                "tpsa": tpsa,
                "hbd": hbd,
                "hba": hba,
                "n_aliphatic_rings": n_aliphatic_rings,
                "n_aromatic_rings": n_aromatic_rings,
                "n_rotatable_bonds": n_rotatable_bonds,
                "sascore": sascore,
                "mscore": mscore,  # Tanimoto-based fallback; overwritten by shape*ESP below
            }

            # Screen against undesirable patterns.
            # Only reject if the input molecule itself doesn't already contain the pattern
            # (matches morpheus.py behaviour).
            reasons = screen_molecule(new_mol)
            if reasons and not ref_has_undesirable:
                mol_entry["rejection_reasons"] = list(dict.fromkeys(reasons))  # dedupe, preserve order
                rejected.append(mol_entry)
            else:
                generated.append(mol_entry)

            # Emit per-molecule generating progress (0.50 → 0.72)
            _gen_progress = 0.50 + (_gen_idx + 1) / max(_n_similar, 1) * 0.22
            yield f"data: {json.dumps({'type': 'progress', 'value': round(_gen_progress, 4), 'stage': 'generating'})}\n\n"

        # Sort generated molecules by mscore descending (highest mscore = rank #1)
        generated.sort(key=lambda m: m.get("mscore") or 0, reverse=True)

        yield f"data: {json.dumps({'type': 'progress', 'value': 0.75, 'stage': 'scoring'})}\n\n"

        # ---- 3D Shape + ESP similarity via espsim ----------------------------
        # Pipeline per molecule pair:
        #   1. Add explicit Hs  (required for accurate 3-D embedding)
        #   2. EmbedAlignScore generates prbNumConfs / refNumConfs 3-D conformers
        #      internally via ETKDGv2, aligns them, and picks the best pair.
        #   3. GetShapeSim + GetEspSim are evaluated on the best-aligned pair.
        if _ESPSIM_AVAILABLE and ref_mol is not None and generated:
            # Probe = input molecule with explicit hydrogens
            prb_mol_esp = Chem.AddHs(Chem.MolFromSmiles(req.original_smiles))

            # Pre-compute reference molecule list with indices
            _esp_ref_mols_h = []
            _esp_valid_indices = []
            for _ei, _entry in enumerate(generated):
                _em = Chem.MolFromSmiles(_entry["smiles"])
                if _em is not None:
                    _esp_ref_mols_h.append(Chem.AddHs(_em))
                    _esp_valid_indices.append(_ei)

            _n_esp = len(_esp_valid_indices)
            if _n_esp > 0:
                # Score one molecule at a time so we can report per-molecule progress
                def _score_one_esp(prb=prb_mol_esp, ref_mol_h=None):
                    try:
                        ss, se = _EmbedAlignScore(
                            prb, [ref_mol_h],
                            renormalize=True,
                            prbNumConfs=10,
                            refNumConfs=10,
                        )
                        return (round(float(ss[0]), 4), round(float(se[0]), 4))
                    except Exception as exc:
                        print(f"ESPsim computation failed: {exc}", flush=True)
                        return None

                for _esp_k, (_esp_idx, _ref_h) in enumerate(zip(_esp_valid_indices, _esp_ref_mols_h)):
                    _esp_result = await loop.run_in_executor(None, _score_one_esp, prb_mol_esp, _ref_h)
                    if _esp_result is not None:
                        ss, se = _esp_result
                        generated[_esp_idx]["shape_sim"] = ss
                        generated[_esp_idx]["esp_sim"] = se
                        generated[_esp_idx]["shape_esp"] = round(ss * se, 4)
                        qed_v = generated[_esp_idx].get("qed")
                        if qed_v is not None:
                            generated[_esp_idx]["mscore"] = round(
                                generated[_esp_idx]["shape_esp"] * float(qed_v), 3
                            )
                    # Emit per-molecule scoring progress (0.75 → 0.95)
                    _esp_progress = 0.75 + (_esp_k + 1) / _n_esp * 0.20
                    yield f"data: {json.dumps({'type': 'progress', 'value': round(_esp_progress, 4), 'stage': 'scoring'})}\n\n"

                # Re-sort with updated shape*ESP-based mscores
                generated.sort(key=lambda m: m.get("mscore") or 0, reverse=True)

        result = {
            "type": "result",
            "similar_fragments": sim_frags,
            "generated_molecules": generated,
            "rejected_molecules": rejected,
            "total_similar": len(similar),
        }
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/validate", response_model=ValidateResponse)
def validate_smiles(req: SmilesRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        return ValidateResponse(valid=False, error="Invalid SMILES string")
    return ValidateResponse(valid=True, canonical=Chem.MolToSmiles(mol))


# ---------------------------------------------------------------------------
# PDB / Protein-Ligand Alignment endpoints
# ---------------------------------------------------------------------------

_EXCLUDE_RESIDUES = {
    'HOH', 'WAT', 'H2O', 'DOD', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN',
    'FE', 'MN', 'CU', 'CO', 'NI', 'CD', 'SO4', 'PO4', 'GOL', 'EDO',
    'ACE', 'NME', 'NH2', 'ACT', 'DMS', 'BME', 'MPD', 'PEG', 'PGE',
    'IOD', 'BR', 'F', 'I', 'NO3', 'SCN',
}


def _ccd_smiles(res_name: str) -> str | None:
    """Fetch the canonical SMILES for a PDB ligand from the RCSB CCD REST API.
    Returns None on any failure so callers can fall back to geometry-based inference."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{res_name.strip().upper()}"
        r = _requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        desc = data.get("rcsb_chem_comp_descriptor", {})
        # Prefer the stereo-aware SMILES, fall back to plain canonical
        smiles = desc.get("smiles_stereo") or desc.get("smiles")
        if smiles:
            mol_check = Chem.MolFromSmiles(smiles)
            if mol_check is not None:
                return smiles
    except Exception:
        pass
    return None


def _draw_ligand_kekulize(mol_2d, size=(300, 300)) -> str:
    """Render a 2-D mol to PNG always using explicit Kekulé bonds (alternating
    single/double in rings, no aromatic-circle notation). Returns base64 PNG."""
    import copy
    m = copy.deepcopy(mol_2d)
    AllChem.Compute2DCoords(m)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        pass
    img = Draw.MolToImage(m, size=size, kekulize=False)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_ligands_from_pdb(pdb_content: str):
    """Parse HETATM records; return list of ligand dicts with correct bond orders."""
    ligand_atoms: dict[tuple, list] = {}
    for line in pdb_content.split('\n'):
        if not line.startswith('HETATM'):
            continue
        try:
            atom_name = line[12:16].strip()
            res_name  = line[17:20].strip()
            chain_id  = line[21]
            res_num   = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            if res_name.upper() not in _EXCLUDE_RESIDUES:
                key = (res_name, chain_id, res_num)
                ligand_atoms.setdefault(key, []).append((atom_name, element, x, y, z))
        except (ValueError, IndexError):
            continue

    results = []
    for (res_name, chain_id, res_num), atoms in ligand_atoms.items():
        if len(atoms) < 3:
            continue

        # ── 1. Rebuild PDB block and get the 3-D mol for the SDF ──────────────
        pdb_lines = []
        for i, (aname, elem, x, y, z) in enumerate(atoms, 1):
            pdb_lines.append(
                f"HETATM{i:5d} {aname:<4s} {res_name:>3s} {chain_id}"
                f"{int(res_num):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {elem:>2s}"
            )
        pdb_lines.append("END")
        pdb_block = '\n'.join(pdb_lines)

        try:
            mol3d = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
            if mol3d is None:
                continue
            # Determine bonds from 3-D geometry (connectivity + bond orders)
            try:
                rdDetermineBonds.DetermineConnectivity(mol3d)
                rdDetermineBonds.DetermineBondOrders(mol3d, charge=0)
            except Exception:
                try:
                    rdDetermineBonds.DetermineBonds(mol3d, charge=0)
                except Exception:
                    pass
            try:
                mol3d.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol3d)
                Chem.SanitizeMol(mol3d)
            except Exception:
                try:
                    Chem.SanitizeMol(mol3d,
                        Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
                except Exception:
                    pass

            sdf_string = Chem.MolToMolBlock(mol3d)

            # ── 2. Get authoritative SMILES: try CCD first, then geometry mol ──
            smiles = _ccd_smiles(res_name)
            if smiles is None:
                try:
                    smiles = Chem.MolToSmiles(mol3d)
                except Exception:
                    smiles = None

            # ── 3. Build a clean mol for 2-D display ──────────────────────────
            mol_2d = Chem.MolFromSmiles(smiles) if smiles else None
            if mol_2d is None:
                mol_2d = Chem.RemoveHs(mol3d)
            if mol_2d is None:
                continue

            img_b64 = _draw_ligand_kekulize(mol_2d, size=(600, 600))

            results.append({
                "ligand_id": f"{res_name}_{chain_id}_{res_num}",
                "res_name": res_name,
                "chain": chain_id,
                "num_atoms": mol3d.GetNumAtoms(),
                "smiles": smiles or "N/A",
                "image": img_b64,
                "sdf": sdf_string,
            })
        except Exception:
            continue
    return results


class FetchPDBRequest(BaseModel):
    pdb_id: str


@app.post("/fetch-pdb")
def fetch_pdb(req: FetchPDBRequest):
    pdb_id = req.pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise HTTPException(status_code=400, detail="PDB IDs must be 4 characters (e.g. 1ATP, 6LU7)")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        resp = _requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Could not fetch PDB ID '{pdb_id}'")
    except _requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error fetching PDB: {e}")
    pdb_content = resp.text
    # Extract basic info
    lines = pdb_content.split('\n')
    atom_count = sum(1 for l in lines if l.startswith('ATOM') or l.startswith('HETATM'))
    hetatm_count = sum(1 for l in lines if l.startswith('HETATM'))
    chains = sorted({l[21] for l in lines if (l.startswith('ATOM') or l.startswith('HETATM')) and len(l) > 21})
    title_parts = [l[10:].strip() for l in lines if l.startswith('TITLE')]
    title = ' '.join(title_parts) if title_parts else 'N/A'
    ligands = _extract_ligands_from_pdb(pdb_content)
    return {
        "pdb_content": pdb_content,
        "pdb_id": pdb_id,
        "title": title[:200],
        "atom_count": atom_count,
        "hetatm_count": hetatm_count,
        "chains": chains,
        "ligands": ligands,
    }


@app.post("/upload-pdb")
async def upload_pdb(file: UploadFile = File(...)):
    content = (await file.read()).decode('utf-8')
    lines = content.split('\n')
    atom_count = sum(1 for l in lines if l.startswith('ATOM') or l.startswith('HETATM'))
    hetatm_count = sum(1 for l in lines if l.startswith('HETATM'))
    chains = sorted({l[21] for l in lines if (l.startswith('ATOM') or l.startswith('HETATM')) and len(l) > 21})
    title_parts = [l[10:].strip() for l in lines if l.startswith('TITLE')]
    title = ' '.join(title_parts) if title_parts else file.filename or 'Uploaded'
    ligands = _extract_ligands_from_pdb(content)
    return {
        "pdb_content": content,
        "pdb_id": file.filename,
        "title": title[:200],
        "atom_count": atom_count,
        "hetatm_count": hetatm_count,
        "chains": chains,
        "ligands": ligands,
    }


class AlignRequest(BaseModel):
    mol_smiles: str
    ligand_sdf: str  # MOL block of the PDB ligand


@app.post("/align-to-ligand")
def align_to_ligand(req: AlignRequest):
    """Align a generated molecule to a PDB ligand using MCS + multi-conformer RMSD."""
    ref_mol = Chem.MolFromMolBlock(req.ligand_sdf, removeHs=False, sanitize=False)
    if ref_mol is None:
        raise HTTPException(status_code=400, detail="Invalid ligand SDF")
    try:
        Chem.SanitizeMol(ref_mol)
    except Exception:
        pass

    query_mol = Chem.MolFromSmiles(req.mol_smiles)
    if query_mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    query_mol = Chem.AddHs(query_mol)

    # Generate conformers
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    cids = rdDistGeom.EmbedMultipleConfs(query_mol, numConfs=50, params=params)
    if len(cids) == 0:
        raise HTTPException(status_code=422, detail="Could not generate 3D conformers")
    AllChem.MMFFOptimizeMoleculeConfs(query_mol, numThreads=0)

    ref_mol_h = Chem.AddHs(ref_mol, addCoords=True)
    mcs = rdFMCS.FindMCS(
        [ref_mol_h, query_mol],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    best_rmsd = float('inf')
    best_conf = 0
    if mcs.numAtoms > 0:
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = ref_mol_h.GetSubstructMatch(mcs_mol)
        if ref_match:
            for cid in cids:
                q_match = query_mol.GetSubstructMatch(mcs_mol)
                if q_match:
                    atom_map = list(zip(q_match, ref_match))
                    rmsd = AllChem.AlignMol(query_mol, ref_mol_h, prbCid=cid, refCid=0, atomMap=atom_map)
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_conf = cid
            q_match = query_mol.GetSubstructMatch(mcs_mol)
            if q_match:
                atom_map = list(zip(q_match, ref_match))
                AllChem.AlignMol(query_mol, ref_mol_h, prbCid=best_conf, refCid=0, atomMap=atom_map)

    # Build single-conformer MOL block
    mol_no_h = Chem.RemoveHs(query_mol)
    fresh = Chem.MolFromSmiles(req.mol_smiles)
    if fresh and mol_no_h.GetNumConformers() > 0:
        try:
            src_conf = mol_no_h.GetConformer(best_conf)
        except Exception:
            src_conf = mol_no_h.GetConformer(0)
        new_conf = Chem.Conformer(fresh.GetNumAtoms())
        for i in range(fresh.GetNumAtoms()):
            pos = src_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        fresh.AddConformer(new_conf, assignId=True)
        aligned_sdf = Chem.MolToMolBlock(fresh, confId=0)
    else:
        aligned_sdf = Chem.MolToMolBlock(mol_no_h, confId=0)

    return {
        "aligned_sdf": aligned_sdf,
        "rmsd": round(best_rmsd, 3) if best_rmsd != float('inf') else None,
        "mcs_atoms": mcs.numAtoms,
        "num_conformers": len(cids),
    }


@app.post("/fragment", response_model=FragmentationResponse)
def fragment_molecule(req: SmilesRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")

    # ---------- molecule info ----------
    rdDepictor.Compute2DCoords(mol)
    mol_img = mol_to_base64_png(mol, size=(600, 600))

    # Generate 3D conformer (same approach as morpheus.py)
    molblock_3d: str | None = None
    try:
        mol_3d = Chem.AddHs(mol)
        result_3d = rdDistGeom.EmbedMolecule(mol_3d, randomSeed=42)
        if result_3d == 0:
            AllChem.MMFFOptimizeMolecule(mol_3d)
            molblock_3d = Chem.MolToMolBlock(mol_3d)
    except Exception:
        pass

    try:
        _input_sascore = round(_sascorer.calculateScore(mol), 2)
    except Exception:
        _input_sascore = None

    mol_info = MoleculeInfo(
        smiles=Chem.MolToSmiles(mol),
        image=mol_img,
        molblock_3d=molblock_3d,
        num_rings=mol.GetRingInfo().NumRings(),
        mw=round(Descriptors.MolWt(mol), 2),
        hbd=Descriptors.NumHDonors(mol),
        hba=Descriptors.NumHAcceptors(mol),
        tpsa=round(Descriptors.TPSA(mol), 2),
        clogp=round(Crippen.MolLogP(mol), 2),
        qed=round(QED.qed(mol), 3),
        sascore=_input_sascore,
    )

    # ---------- fragmentation ----------
    decomposition = decompose_molecule_with_wildcards(
        mol,
        include_terminal_substituents=True,
        preserve_fused_rings=True,
        max_terminal_atoms=max(3, min(5, req.max_terminal_atoms)),
    )

    all_frags = decomposition["rings"] + decomposition["non_rings"]
    total = len(all_frags)
    all_frag_smiles = [frag["wildcard_smiles"] for frag in all_frags]

    # Only display fragments with >= 3 heavy atoms (same logic as morpheus.py)
    displayable: list[FragmentInfo] = []
    for i, frag in enumerate(all_frags):
        wsmi = frag["wildcard_smiles"]
        if count_heavy_atoms(wsmi) < 3:
            continue

        frag_mol = frag.get("frag_mol")
        if frag_mol is None:
            frag_mol = Chem.MolFromSmiles(wsmi)
        if frag_mol is None:
            continue

        displayable.append(
            FragmentInfo(
                wildcard_smiles=wsmi,
                base_smiles=frag["base_smiles"],
                frag_type=frag["frag_type"],
                size=frag["size"],
                hetero_count=frag["hetero_count"],
                image=mol_to_base64_png(frag_mol, size=(300, 300)),
                all_fragment_index=i,
            )
        )

    return FragmentationResponse(
        molecule=mol_info,
        fragments=displayable,
        all_fragment_smiles=all_frag_smiles,
        total_fragments=total,
        displayed_fragments=len(displayable),
    )


# ============================================================================
# Substructure filtering
# ============================================================================

class SubstructureMatchRequest(BaseModel):
    pattern: str
    smiles_list: list[str]


@app.post("/substructure-match")
def substructure_match(req: SubstructureMatchRequest):
    """Return the subset of smiles_list that contains the given SMARTS/SMILES pattern."""
    # Try as SMARTS first, then as SMILES
    query = Chem.MolFromSmarts(req.pattern)
    if query is None:
        query = Chem.MolFromSmiles(req.pattern)
    if query is None:
        raise HTTPException(status_code=400, detail="Invalid SMARTS/SMILES pattern")
    matching: list[str] = []
    for smi in req.smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol and mol.HasSubstructMatch(query):
            matching.append(smi)
    return {"matching_smiles": matching}


# ============================================================================
# SDF / SMILES export
# ============================================================================

class ExportMolecule(BaseModel):
    smiles: str
    id: int = 0
    new_fragment_smiles: str = ""
    frag_similarity: float | None = None
    mol_similarity: float | None = None
    mw: float | None = None
    clogp: float | None = None
    qed: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    n_aliphatic_rings: int | None = None
    n_aromatic_rings: int | None = None
    n_rotatable_bonds: int | None = None
    sascore: float | None = None
    mscore: float | None = None


class ExportRequest(BaseModel):
    molecules: list[ExportMolecule]


@app.post("/export-sdf")
def export_sdf(req: ExportRequest):
    """Accept a list of molecule records and return an SDF file."""
    buf = io.StringIO()
    writer = Chem.SDWriter(buf)
    prop_order = [
        "id", "new_fragment_smiles", "frag_similarity", "mol_similarity",
        "mw", "clogp", "qed", "tpsa", "hbd", "hba",
        "n_aliphatic_rings", "n_aromatic_rings", "n_rotatable_bonds",
        "sascore", "mscore",
    ]
    for entry in req.molecules:
        mol = Chem.MolFromSmiles(entry.smiles)
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mol.SetProp("_Name", f"mol_{entry.id}")
        for prop in prop_order:
            val = getattr(entry, prop, None)
            if val is not None:
                mol.SetProp(prop, str(val))
        writer.write(mol)
    writer.close()
    sdf_bytes = buf.getvalue().encode()
    return Response(
        content=sdf_bytes,
        media_type="chemical/x-mdl-sdfile",
        headers={"Content-Disposition": 'attachment; filename="molecules.sdf"'},
    )


# ============================================================================
# Retrosynthetic planning – auto‑launching proxy to synth_planner (port 8001)
# ============================================================================
import subprocess, time, threading, atexit

SYNTH_PLANNER_URL = "http://127.0.0.1:8001"
BACKEND_DIR = str(Path(__file__).parent)


def _find_synplanner_python() -> str:
    """Locate the Python executable for the 'synplanner' conda environment.

    Resolution order:
    1. SYNPLANNER_PYTHON environment variable.
    2. Ask conda where the env lives.
    3. Well-known install locations (~/miniconda3, ~/anaconda3, etc.).
    Falls back to bare 'python' with a warning if nothing is found.
    """
    override = os.environ.get("SYNPLANNER_PYTHON", "").strip()
    if override and Path(override).exists():
        return override

    # Try asking conda
    for conda_cmd in ["conda", str(Path.home() / "miniconda3/bin/conda"),
                      str(Path.home() / "anaconda3/bin/conda")]:
        try:
            r = subprocess.run(
                [conda_cmd, "run", "-n", "synplanner", "python",
                 "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                p = r.stdout.strip().splitlines()[-1]
                if p and Path(p).exists():
                    return p
        except Exception:
            pass

    # Scan common install roots
    for root in [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "opt" / "miniconda3",
        Path.home() / "opt" / "anaconda3",
        Path("/opt/homebrew/anaconda3"),
        Path("/opt/homebrew/miniconda3"),
    ]:
        candidate = root / "envs" / "synplanner" / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    print(
        "[main] WARNING: Could not locate the 'synplanner' conda env Python.\n"
        "  Set the SYNPLANNER_PYTHON env var to the full path, e.g.:\n"
        "  export SYNPLANNER_PYTHON=~/miniconda3/envs/synplanner/bin/python\n"
        "  Retrosynthetic planning will be unavailable until this is resolved."
    )
    return "python"  # last-resort fallback


SYNPLANNER_PYTHON = _find_synplanner_python()

_synth_proc: subprocess.Popen | None = None
_synth_lock = threading.Lock()


def _is_synth_planner_up() -> bool:
    """Return True if the synth_planner service responds to a health check."""
    try:
        r = _requests.get(f"{SYNTH_PLANNER_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _is_synth_planner_initialized() -> bool:
    """Return True if building blocks are fully loaded."""
    try:
        r = _requests.get(f"{SYNTH_PLANNER_URL}/health", timeout=3)
        return r.json().get("initialized", False)
    except Exception:
        return False


def _start_synth_planner() -> bool:
    """Launch synth_planner:app on port 8001 in the synplanner conda env.
    Returns True once the health endpoint responds (server is up)."""
    global _synth_proc
    with _synth_lock:
        # Double-check inside lock
        if _is_synth_planner_up():
            return True
        # Kill any stale process holding port 8001 before starting a fresh one
        try:
            stale = subprocess.run(
                ["lsof", "-ti", "tcp:8001"],
                capture_output=True, text=True,
            ).stdout.strip()
            if stale:
                for pid in stale.splitlines():
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], check=False)
                    except Exception:
                        pass
                time.sleep(0.5)
        except Exception:
            pass
        print("[main] Starting synth_planner service on port 8001 …")
        _synth_proc = subprocess.Popen(
            [
                SYNPLANNER_PYTHON, "-m", "uvicorn",
                "synth_planner:app", "--host", "127.0.0.1", "--port", "8001",
            ],
            cwd=BACKEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait up to 30 s for the server to start accepting TCP
        for _ in range(60):
            time.sleep(0.5)
            if _synth_proc.poll() is not None:
                print("[main] synth_planner process exited prematurely")
                return False
            if _is_synth_planner_up():
                print("[main] synth_planner is up")
                return True
        print("[main] synth_planner did not start within 30 s")
        return False


def _wait_for_synth_init(timeout: int = 600) -> tuple[bool, str]:
    """Poll health until building blocks are loaded (up to *timeout* seconds).

    Returns (True, "") on success, or (False, error_message) on failure / timeout.
    Fails immediately if the planner reports an init_error instead of waiting.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(f"{SYNTH_PLANNER_URL}/health", timeout=3)
            health = r.json()
            if health.get("initialized", False):
                return True, ""
            # Fail fast: initialisation encountered an error, stop polling
            err = health.get("init_error")
            if err and not health.get("initializing", False):
                return False, err
        except Exception:
            pass
        time.sleep(2)
    return False, "Planner initialisation timed out (building blocks still loading)."


@atexit.register
def _kill_synth_planner():
    global _synth_proc
    if _synth_proc is not None and _synth_proc.poll() is None:
        _synth_proc.terminate()


class PlanSynthesisRequest(BaseModel):
    smiles: str
    max_routes: int = 5
    max_depth: int = 4
    max_iterations: int = 200
    min_mol_size: int = 1


@app.post("/plan-synthesis")
async def proxy_plan_synthesis(req: PlanSynthesisRequest):
    """Auto-start synth_planner if needed, wait for init, then forward."""
    from fastapi.responses import JSONResponse as _JSONResponse

    # 1. Make sure the service is running (auto-start) — runs in thread pool
    #    so time.sleep inside _start_synth_planner doesn't block the event loop.
    loop = asyncio.get_event_loop()
    if not _is_synth_planner_up():
        started = await loop.run_in_executor(None, _start_synth_planner)
        if not started:
            return _JSONResponse(
                status_code=503,
                content={"detail": "Could not start the retrosynthetic planner service. "
                                   "Check that the synplanner conda env exists."},
            )

    # 2. Wait until building blocks are loaded (first-time can take minutes)
    if not _is_synth_planner_initialized():
        ok, init_err = await loop.run_in_executor(None, lambda: _wait_for_synth_init(timeout=600))
        if not ok:
            return _JSONResponse(
                status_code=503,
                content={"detail": f"Planner initialisation failed: {init_err}"},
            )

    # 3. Forward the request via httpx (non-blocking async HTTP)
    try:
        async with _httpx.AsyncClient(timeout=660.0) as client:
            resp = await client.post(
                f"{SYNTH_PLANNER_URL}/plan-synthesis",
                json=req.model_dump(),
            )
    except _httpx.ConnectError:
        return _JSONResponse(
            status_code=503,
            content={"detail": "Planner service connection lost. It may have crashed — check terminal logs."},
        )
    except _httpx.TimeoutException:
        return _JSONResponse(
            status_code=504,
            content={"detail": "Retrosynthetic planning timed out (limit: 660 s)."},
        )
    except BaseException as exc:
        traceback.print_exc()
        return _JSONResponse(
            status_code=500,
            content={"detail": f"Proxy error: {exc}"},
        )

    # 4. Always return JSON — even if synth_planner returned a plain-text error page
    if not resp.content:
        return _JSONResponse(
            status_code=502,
            content={"detail": f"Planner returned an empty response (HTTP {resp.status_code}). "
                               "The service may have crashed — check terminal logs."},
        )
    try:
        data = resp.json()
    except Exception:
        return _JSONResponse(
            status_code=502,
            content={"detail": f"Planner returned non-JSON (HTTP {resp.status_code}): {resp.text[:400]}"},
        )
    return _JSONResponse(content=data, status_code=resp.status_code)


@app.get("/synth-planner-health")
def synth_planner_health():
    """Check if the synth_planner micro-service is reachable."""
    try:
        resp = _requests.get(f"{SYNTH_PLANNER_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unavailable"}

