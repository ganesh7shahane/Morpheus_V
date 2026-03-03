# Morpheus V

A cheminformatics tool for bioisosteric and R-group replacement, built with React + Vite on the frontend and FastAPI + RDKit on the backend.

**Features:**
- Molecule fragmentation & 2D/3D visualisation
- Fragment-based search & replace with real-time progress (SSE)
- Generated molecule grid with parallel-coordinates filter, sorting, and pagination
- Shape + ESP similarity scoring via espsim
- Protein-Ligand alignment with Mol* 3D viewer
- Retrosynthetic planning via MCTS (requires optional `synplanner` conda env)

---

## Quick start (new machine)

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Node.js | ≥ 18 | [nodejs.org](https://nodejs.org) |
| Python | ≥ 3.10 | system, pyenv, or conda |

### Steps

```bash
git clone https://github.com/<your-org>/Morpheus_V.git
cd Morpheus_V
npm install
npm run dev
```

**That's it.** On the very first run, `npm run dev` will:

1. Start the Vite frontend dev server immediately (port 5173).
2. Detect Python 3.10+ on your machine.
3. Create an isolated virtual environment at `backend/.venv`.
4. Install all Python dependencies automatically (fastapi, uvicorn, rdkit, espsim, …).
5. Start the FastAPI backend (port 8000).

Subsequent `npm run dev` invocations skip the install step unless `requirements.txt` has changed.

| Server | URL |
|---|---|
| React / Vite frontend | http://localhost:5173 |
| FastAPI backend | http://127.0.0.1:8000 |

> **No activation needed.** The launcher (`backend/start_backend.sh`) manages its own `backend/.venv` and does not modify your global Python environment.

---

## Retrosynthetic Planning (optional feature)

Requires a separate `synplanner` conda environment and ~2 GB of model weights.

### Create the synplanner env

```bash
conda create -n synplanner python=3.10
conda activate synplanner
pip install synplanner cgrtools fastapi "uvicorn[standard]"
```

### Download model weights

```bash
conda activate synplanner
python -c "from synplan.utils.loading import download_all_data; download_all_data('synplan_data')"
```

Creates a `synplan_data/` folder at the repo root (~2 GB).

### Path configuration (only if auto-detection fails)

The backend auto-detects the `synplanner` Python in common locations.
If detection fails a warning appears in the backend console.
Override via an env variable:

```bash
export SYNPLANNER_PYTHON=~/miniconda3/envs/synplanner/bin/python
```

Once set up, clicking **"Run Retrosynthetic Planning"** auto-starts the secondary service on port 8001. The first run loads 311 K building blocks and takes 3–5 min; subsequent runs are fast.

---

## Data files

| File | Size | Tracked in git |
|---|---|---|
| `data/building_blocks_em_sa_ln_with_ids.sdf.gz` | 25 MB | yes |
| `data/fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz` | 5 MB | yes |
| `data/*.sdf` (uncompressed, 276 MB) | 276 MB | no — too large |
| `data/*.pickle` (derived cache) | ~30 MB | no — auto-generated |
| `synplan_data/` (model weights) | ~2 GB | no — download separately |

---

## Architecture

```
npm run dev
├── dev:frontend  →  Vite dev server              (port 5173)
└── dev:backend   →  backend/start_backend.sh
                       ├── Creates backend/.venv if needed
                       ├── pip installs requirements.txt if needed
                       └── uvicorn main:app         (port 8000)
                                  └── spawned on demand:
                                      uvicorn synth_planner:app (port 8001, synplanner env)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Could not find Python 3.10+` | Install Python via `brew install python` or `conda install python=3.12` |
| Install fails / broken venv | Delete `backend/.venv`, then re-run `npm run dev` |
| `ModuleNotFoundError` after install | Delete `backend/.venv` and `backend/.venv/.deps_ok`, then re-run |
| Retrosynthesis spins / "planner service" error | Check backend console for `WARNING: Could not locate synplanner Python` and set `SYNPLANNER_PYTHON` |
| First retrosynthesis call times out | Normal — building block loading takes 3–5 min on first run |
| `data/*.sdf` missing after clone | Expected — the 276 MB file is gitignored; the backend uses the `.sdf.gz` directly |

