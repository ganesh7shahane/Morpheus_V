import React, { useState, useCallback, useEffect, useRef, useMemo, lazy, Suspense } from "react";
import {
  Container,
  Typography,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Box,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Paper,
  Table,
  TableBody,
  TableRow,
  TableCell,
  IconButton,
  Tooltip,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Slider,
  LinearProgress,
  Pagination,
  Menu,
  Switch,
  Checkbox,
  FormControlLabel,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ScienceIcon from "@mui/icons-material/Science";
import EditIcon from "@mui/icons-material/Edit";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import LightModeIcon from "@mui/icons-material/LightMode";
import SearchIcon from "@mui/icons-material/Search";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CheckIcon from "@mui/icons-material/Check";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DownloadIcon from "@mui/icons-material/Download";
import ClearIcon from "@mui/icons-material/Clear";
import BiotechIcon from "@mui/icons-material/Biotech";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import axios from "axios";
import Plot from "react-plotly.js";
import ErrorBoundary from "./ErrorBoundary";

// Lazy-load the heavy Ketcher bundle — only fetched when the accordion is opened
const KetcherEditor = lazy(() => import("./KetcherEditor"));

const API_URL = "http://127.0.0.1:8000";

// ---- Fragment highlight image component ----
// Example molecules from morpheus.py
const EXAMPLE_MOLECULES: Record<string, string> = {
  "": "",
  AZ20: "C[C@@H]1COCCN1C2=NC(=NC(=C2)C3(CC3)[S@](=O)(=O)C)C4=CN=CC5=C4C=CN5",
  Imatinib:
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
  Rofecoxib: "CS(=O)(=O)C1=CC=C(C2=C(C3=CC=CC=C3)C(=O)OC2)C=C1",
  Gefitinib: "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
  Ibrutinib:
    "C=CC(=O)N1CCC[C@H](C1)N2C3=NC=NC(=C3C(=N2)C4=CC=C(C=C4)OC5=CC=CC=C5)N",
  Acalabrutinib:
    "CC#CC(=O)N1CCC[C@H]1C2=NC(=C3N2C=CN=C3N)C4=CC=C(C=C4)C(=O)NC5=CC=CC=N5",
  Dasatinib:
    "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO",
  Maraviroc:
    "CC1=NN=C(N1C2C[C@H]3CC[C@@H](C2)N3CC[C@@H](C4=CC=CC=C4)NC(=O)C5CCC(CC5)(F)F)C(C)C",
  Roniciclib:
    "C[C@H]([C@@H](C)OC1=NC(=NC=C1C(F)(F)F)NC2=CC=C(C=C2)[S@](=N)(=O)C3CC3)O",
  GV134:
    "FC1(F)CN(C1)C(=O)C=2N(C)c3cc(ccc3C2)c4nccc(n4)N5CC[C@@H](C5)C=6C=NNC6",
};

interface FragmentInfo {
  wildcard_smiles: string;
  base_smiles: string;
  frag_type: string;
  size: number;
  hetero_count: number;
  image: string; // base64
  all_fragment_index: number; // index in the full fragment list (for reassembly)
}

interface MoleculeInfo {
  smiles: string;
  image: string;
  molblock_3d: string | null;
  num_rings: number;
  mw: number;
  hbd: number;
  hba: number;
  tpsa: number;
  clogp: number;
  qed: number;
  sascore: number | null;
}

interface SimilarFragment {
  smiles: string;
  similarity: number;
  image: string;
  library: string;
}

interface FragmentationResult {
  molecule: MoleculeInfo;
  fragments: FragmentInfo[];
  all_fragment_smiles: string[]; // ALL fragments (incl. small) for reassembly
  total_fragments: number;
  displayed_fragments: number;
}

interface PatternInfo {
  smarts: string;
  name: string;
  image: string;
}

interface GeneratedMolecule {
  id: number;
  smiles: string;
  image: string;
  new_fragment_smiles?: string;
  frag_library?: string;
  frag_library_idx?: number;
  frag_similarity: number;
  mol_similarity: number | null;
  shape_sim: number | null;
  esp_sim: number | null;
  shape_esp: number | null;
  mw: number | null;
  clogp: number | null;
  qed: number | null;
  tpsa: number | null;
  hbd: number | null;
  hba: number | null;
  n_aliphatic_rings: number | null;
  n_aromatic_rings: number | null;
  n_rotatable_bonds: number | null;
  sascore: number | null;
  mscore: number | null;
}

interface RejectedMolecule extends GeneratedMolecule {
  rejection_reasons: string[];
}

interface FilterRange {
  min: number;
  max: number;
  currentMin: number;
  currentMax: number;
}

interface PDBLigand {
  ligand_id: string;
  res_name: string;
  chain: string;
  num_atoms: number;
  smiles: string;
  image: string;   // base64
  sdf: string;     // MOL block
}

interface PDBInfo {
  pdb_content: string;
  pdb_id: string;
  title: string;
  atom_count: number;
  hetatm_count: number;
  chains: string[];
  ligands: PDBLigand[];
}

interface AlignmentResult {
  aligned_sdf: string;
  rmsd: number | null;
  mcs_atoms: number;
  num_conformers: number;
}

// ----- Retrosynthesis types -----
interface BuildingBlock {
  smiles: string;
  id: string | null;
}

interface RouteIntermediate {
  smiles: string;
}

interface SynthRoute {
  node_id: number;
  score: number;
  num_steps: number;
  svg: string | null;
  svg_error?: string;
  building_blocks: BuildingBlock[];
  intermediates: RouteIntermediate[];
}

interface RetroResult {
  success: boolean;
  solved: boolean;
  routes: SynthRoute[];
  error?: string;
  target_smiles?: string;
}

// Descriptors shown in parallel coordinates (order = axis order)
const DESCRIPTOR_KEYS: { key: string; label: string }[] = [
  { key: "mscore", label: "MScore" },
  { key: "shape_esp", label: "Shape×ESP" },
  { key: "shape_sim", label: "Shape Sim" },
  { key: "esp_sim", label: "ESP Sim" },
  { key: "mol_similarity", label: "Tanimoto" },
  { key: "frag_similarity", label: "Frag. Sim" },
  { key: "frag_library_idx", label: "Library" },
  { key: "mw", label: "MW" },
  { key: "qed", label: "QED" },
  { key: "clogp", label: "cLogP" },
  { key: "tpsa", label: "TPSA" },
  { key: "hbd", label: "HBD" },
  { key: "hba", label: "HBA" },
  { key: "n_aliphatic_rings", label: "Ali. Rings" },
  { key: "n_aromatic_rings", label: "Aro. Rings" },
  { key: "n_rotatable_bonds", label: "Rot. Bonds" },
  { key: "sascore", label: "SA Score" },
];

function App() {
  const [smilesInput, setSmilesInput] = useState("");
  const [selectedExample, setSelectedExample] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FragmentationResult | null>(null);
  const [maxTerminalAtoms, setMaxTerminalAtoms] = useState(3);
  const [selectedFragIdx, setSelectedFragIdx] = useState<number | null>(null);
  const [ketcherExpanded, setKetcherExpanded] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  // Search & Replace state
  const [simThreshold, setSimThreshold] = useState(0.3);
  const [maxResults, setMaxResults] = useState(200);
  const [searching, setSearching] = useState(false);
  const [searchProgress, setSearchProgress] = useState(0);
  const [searchStage, setSearchStage] = useState<string>("searching");
  const [searchError, setSearchError] = useState<string | null>(null);
  const [generatedMols, setGeneratedMols] = useState<GeneratedMolecule[]>([]);
  const [rejectedMols, setRejectedMols] = useState<RejectedMolecule[]>([]);
  const [totalSimilar, setTotalSimilar] = useState(0);
  const [similarFragments, setSimilarFragments] = useState<SimilarFragment[]>([]);
  const [libIndexLabels, setLibIndexLabels] = useState<string[]>([]);
  const [molPage, setMolPage] = useState(1);
  const MOLS_PER_PAGE = 20;
  const [molFilters, setMolFilters] = useState<Record<string, FilterRange>>({});
  const [showAllMols, setShowAllMols] = useState(false);
  const [legendCols, setLegendCols] = useState<string[]>(["mscore", "mol_similarity", "qed"]);
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<string>("mscore");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selectedMolIds, setSelectedMolIds] = useState<Set<number>>(new Set());
  const [molGridMenuAnchor, setMolGridMenuAnchor] = useState<null | HTMLElement>(null);
  const [substructInput, setSubstructInput] = useState("");
  const [substructMatchSmiles, setSubstructMatchSmiles] = useState<Set<string> | null>(null);
  const [substructExclude, setSubstructExclude] = useState(false);
  const [substructError, setSubstructError] = useState<string | null>(null);
  const [patternData, setPatternData] = useState<PatternInfo[]>([]);

  // Fragment library selector
  const [fragmentLibraries, setFragmentLibraries] = useState<string[]>([]);
  const [selectedLibraries, setSelectedLibraries] = useState<Set<string>>(new Set());

  // Protein-Ligand Alignment state
  const [pdbInfo, setPdbInfo] = useState<PDBInfo | null>(null);
  const [pdbIdInput, setPdbIdInput] = useState("");
  const [fetchingPdb, setFetchingPdb] = useState(false);
  const [pdbError, setPdbError] = useState<string | null>(null);
  const [selectedAlignMol, setSelectedAlignMol] = useState<string>("");
  const [selectedLigandId, setSelectedLigandId] = useState<string>("");
  const [aligning, setAligning] = useState(false);
  const [alignmentResult, setAlignmentResult] = useState<AlignmentResult | null>(null);
  const [alignError, setAlignError] = useState<string | null>(null);
  const molstarIframeRef = useRef<HTMLIFrameElement>(null);
  const [bgColor, setBgColor] = useState("#ffffff");

  // Retrosynthetic Planning state
  const [retroSelectedSmiles, setRetroSelectedSmiles] = useState<string>("");
  const [retroMaxDepth, setRetroMaxDepth] = useState(4);
  const [retroMaxIter, setRetroMaxIter] = useState(200);
  const [retroMinMolSize, setRetroMinMolSize] = useState(1);
  const [retroNumRoutes, setRetroNumRoutes] = useState(5);
  const [retroRunning, setRetroRunning] = useState(false);
  const [retroResult, setRetroResult] = useState<RetroResult | null>(null);
  const [retroError, setRetroError] = useState<string | null>(null);
  const [retroCopiedSmiles, setRetroCopiedSmiles] = useState<string | null>(null);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? "dark" : "light",
          ...(darkMode
            ? {
                background: { default: "#0f1117", paper: "#1a1d27" },
                primary: { main: "#5c9eff" },
              }
            : {
                background: { default: "#edf0f7", paper: "#f5f8ff" },
                primary: { main: "#2563eb" },
                secondary: { main: "#7c3aed" },
                divider: "#d4dcea",
                text: {
                  primary: "#1e2a3a",
                  secondary: "#55677d",
                },
                action: {
                  hover: "rgba(37,99,235,0.06)",
                  selected: "rgba(37,99,235,0.10)",
                },
              }),
        },
        components: {
          MuiPaper: {
            styleOverrides: {
              root: {
                borderRadius: 16,
                ...(!darkMode && {
                  backgroundImage: "none",
                  boxShadow: "0 1px 4px 0 rgba(30,42,58,0.07), 0 0 0 1px rgba(30,42,58,0.05)",
                }),
              },
            },
          },
          MuiChip: {
            styleOverrides: {
              root: {
                fontWeight: 500,
              },
            },
          },
          MuiDivider: {
            styleOverrides: {
              root: {
                ...(!darkMode && { borderColor: "#d4dcea" }),
              },
            },
          },
        },
      }),
    [darkMode]
  );

  // Handle example selection
  // Fragment molecule via backend
  const handleFragment = useCallback(async (overrideSmiles?: string) => {
    const smiles = (overrideSmiles ?? smilesInput).trim();
    if (!smiles) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedFragIdx(null);
    setGeneratedMols([]);
    setRejectedMols([]);
    setTotalSimilar(0);
    setSearchError(null);
    setMolFilters({});
    setShowAllMols(false);
    setLegendCols(["mscore", "mol_similarity", "qed"]);
    setSortKey("mscore");
    setSortDir("desc");

    try {
      const resp = await axios.post<FragmentationResult>(
        `${API_URL}/fragment`,
        { smiles, max_terminal_atoms: maxTerminalAtoms }
      );
      setResult(resp.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError("Failed to fragment molecule. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  }, [smilesInput, maxTerminalAtoms]);

  const handleExampleChange = (value: string) => {
    setSelectedExample(value);
    if (value && EXAMPLE_MOLECULES[value]) {
      const smiles = EXAMPLE_MOLECULES[value];
      setSmilesInput(smiles);
      handleFragment(smiles);
    }
  };

  // Re-fragment automatically when max_terminal_atoms changes (if a molecule is already loaded)
  useEffect(() => {
    if (smilesInput.trim()) {
      handleFragment();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maxTerminalAtoms]);

  // Fetch undesirable patterns once on mount
  useEffect(() => {
    fetch(`${API_URL}/undesirable-patterns`)
      .then((r) => r.json())
      .then(setPatternData)
      .catch(() => {});
  }, []);

  // Fetch available fragment libraries once on mount
  useEffect(() => {
    fetch(`${API_URL}/fragment-libraries`)
      .then((r) => r.json())
      .then((data) => {
        const libs: string[] = data.libraries ?? [];
        setFragmentLibraries(libs);
        const DEFAULT_LIBS = ["VeryCommon.txt.gz", "Common.txt.gz", "LessCommon.txt.gz", "Rare.txt.gz"];
        setSelectedLibraries(new Set(libs.filter((l) => DEFAULT_LIBS.includes(l))));
      })
      .catch(() => {});
  }, []);

  // Load 3Dmol.js script once
  useEffect(() => {
    if (document.getElementById("3dmol-script")) return;
    const script = document.createElement("script");
    script.id = "3dmol-script";
    script.src = "https://3Dmol.csb.pitt.edu/build/3Dmol-min.js";
    script.async = true;
    document.head.appendChild(script);
  }, []);

  // Render 3D viewer when result changes
  useEffect(() => {
    if (!result?.molecule?.molblock_3d || !viewerContainerRef.current) return;

    const render3D = () => {
      const $3Dmol = (window as any).$3Dmol;
      if (!$3Dmol) {
        // Script not loaded yet, retry
        setTimeout(render3D, 200);
        return;
      }
      // Clear previous viewer
      if (viewerRef.current) {
        try { viewerRef.current.clear(); } catch { /* ignore */ }
      }
      viewerContainerRef.current!.innerHTML = "";

      const viewer = $3Dmol.createViewer(viewerContainerRef.current, {
        backgroundColor: darkMode ? "#1a1d27" : "white",
      });
      viewer.addModel(result.molecule.molblock_3d, "sdf");
      // Style all heavy atoms with sticks
      viewer.setStyle({}, { stick: { radius: 0.2 } });
      // Hide all H first, then show only polar H (bonded to N, O, S, P, or halogen)
      viewer.setStyle({ elem: "H" }, {});
      const atoms: any[] = viewer.getModel().selectedAtoms({});
      const atomByIndex: Record<number, any> = {};
      atoms.forEach((a: any) => { atomByIndex[a.index] = a; });
      const polarHSerials: number[] = [];
      atoms.forEach((a: any) => {
        if (a.elem === "H" && Array.isArray(a.bonds)) {
          const isPolar = a.bonds.some((bIdx: number) => {
            const nbr = atomByIndex[bIdx];
            return nbr && nbr.elem !== "C" && nbr.elem !== "H";
          });
          if (isPolar) polarHSerials.push(a.serial);
        }
      });
      if (polarHSerials.length > 0) {
        viewer.setStyle({ serial: polarHSerials }, { stick: { radius: 0.12 } });
      }
      viewer.zoomTo();
      viewer.render();
      viewerRef.current = viewer;
    };

    render3D();
  }, [result, darkMode]);

  // Search & Replace via SSE
  const handleSearchReplace = useCallback(async () => {
    if (selectedFragIdx === null || !result) return;
    if (selectedLibraries.size === 0) return;

    setSearching(true);
    setSearchProgress(0);
    setSearchStage("searching");
    setSearchError(null);
    setGeneratedMols([]);
    setRejectedMols([]);
    setTotalSimilar(0);
    setSimilarFragments([]);
    setMolFilters({});
    setShowAllMols(false);
    setLegendCols(["mscore", "mol_similarity", "qed"]);
    setSortKey("mscore");
    setSortDir("desc");

    try {
      const selectedFrag = result.fragments[selectedFragIdx];
      const body = {
        query_smiles: selectedFrag.wildcard_smiles,
        // Send ALL fragments (not just displayed) so reassembly works correctly,
        // and use the fragment's true index within that full list.
        all_fragment_smiles: result.all_fragment_smiles,
        selected_index: selectedFrag.all_fragment_index,
        original_smiles: result.molecule.smiles,
        similarity_threshold: simThreshold,
        top_n: maxResults,
        library_names: [...selectedLibraries],
      };

      const resp = await fetch(`${API_URL}/search-replace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!resp.ok || !resp.body) throw new Error("Server error");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.replace(/^data: /, "").trim();
          if (!line) continue;
          try {
            const evt = JSON.parse(line);
            if (evt.type === "progress") {
              setSearchProgress(evt.value);
              if (evt.stage) setSearchStage(evt.stage);
            } else if (evt.type === "result") {
              // Assign sequential IDs (1 = highest similarity, already sorted by backend)
              // Build library → numeric-index mapping for categorical parcoords axis
              const LIB_DISPLAY: Record<string, string> = {
                "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz": "ChEMBL",
                "VeryCommon.txt.gz": "Very Common",
                "Common.txt.gz": "Common",
                "LessCommon.txt.gz": "Less Common",
                "Rare.txt.gz": "Rare",
                "VeryRare.txt.gz": "Very Rare",
                "ExtremelyRare.txt.gz": "Ext. Rare",
                "UltraRare.txt.gz": "Ultra Rare",
                "Doubletons.txt.gz": "Doubletons",
                "Singletons.txt.gz": "Singletons",
              };
              const LIB_ORDER = [
                "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz",
                "VeryCommon.txt.gz", "Common.txt.gz", "LessCommon.txt.gz",
                "Rare.txt.gz", "VeryRare.txt.gz", "ExtremelyRare.txt.gz", "UltraRare.txt.gz",
                "Doubletons.txt.gz", "Singletons.txt.gz",
              ];
              const rawLibs: string[] = [
                ...new Set(
                  (evt.generated_molecules as any[])
                    .map((m: any) => m.frag_library as string)
                    .filter(Boolean)
                ),
              ].sort((a, b) => {
                const ia = LIB_ORDER.indexOf(a), ib = LIB_ORDER.indexOf(b);
                return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
              });
              const libToIdx: Record<string, number> = {};
              rawLibs.forEach((lib, i) => { libToIdx[lib] = i; });
              const shortLabels = rawLibs.map(
                (lib) => LIB_DISPLAY[lib] ?? lib.replace(/\.txt\.gz$/, "")
              );
              setLibIndexLabels(shortLabels);
              const mols: GeneratedMolecule[] = (evt.generated_molecules as any[]).map(
                (m: any, i: number) => ({
                  ...m,
                  id: i + 1,
                  frag_library_idx: m.frag_library != null ? (libToIdx[m.frag_library] ?? 0) : 0,
                })
              );
              setGeneratedMols(mols);
              // Build parallel-coordinates filter ranges from actual data
              const filters: Record<string, FilterRange> = {};
              for (const { key } of DESCRIPTOR_KEYS) {
                const values = mols
                  .map((m) => (m as any)[key])
                  .filter((v) => v != null) as number[];
                if (values.length > 0) {
                  const min = Math.min(...values);
                  const max = Math.max(...values);
                  filters[key] = { min, max, currentMin: min, currentMax: max };
                }
              }
              setMolFilters(filters);
              setRejectedMols(evt.rejected_molecules ?? []);
              setTotalSimilar(evt.total_similar);
              setSimilarFragments(evt.similar_fragments ?? []);
              setMolPage(1);
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (e: any) {
      setSearchError(e.message || "Search failed");
    } finally {
      setSearching(false);
      setSearchProgress(1);
    }
  }, [selectedFragIdx, result, simThreshold, maxResults, selectedLibraries]);

  // Molecules visible in the grid — apply parallel-coordinates filter ranges
  const filteredMols = useMemo(() => {
    if (showAllMols || Object.keys(molFilters).length === 0) return generatedMols;
    return generatedMols.filter((mol) => {
      for (const { key } of DESCRIPTOR_KEYS) {
        const filter = molFilters[key];
        if (!filter) continue;
        const val = (mol as any)[key];
        if (val != null && (val < filter.currentMin || val > filter.currentMax)) return false;
      }
      return true;
    });
  }, [generatedMols, molFilters, showAllMols]);

  // ---- Protein-Ligand Alignment handlers ----
  const handleFetchPdb = useCallback(async () => {
    const id = pdbIdInput.trim();
    if (!id) return;
    setFetchingPdb(true);
    setPdbError(null);
    setPdbInfo(null);
    setAlignmentResult(null);
    setAlignError(null);
    setSelectedLigandId("");
    setSelectedAlignMol("");
    try {
      const resp = await axios.post<PDBInfo>(`${API_URL}/fetch-pdb`, { pdb_id: id });
      setPdbInfo(resp.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setPdbError(err.response.data.detail);
      } else {
        setPdbError("Failed to fetch PDB. Is the backend running?");
      }
    } finally {
      setFetchingPdb(false);
    }
  }, [pdbIdInput]);

  const handleUploadPdb = useCallback(async (file: File) => {
    setFetchingPdb(true);
    setPdbError(null);
    setPdbInfo(null);
    setAlignmentResult(null);
    setAlignError(null);
    setSelectedLigandId("");
    setSelectedAlignMol("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const resp = await axios.post<PDBInfo>(`${API_URL}/upload-pdb`, formData);
      setPdbInfo(resp.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setPdbError(err.response.data.detail);
      } else {
        setPdbError("Failed to upload PDB file.");
      }
    } finally {
      setFetchingPdb(false);
    }
  }, []);

  const handleAlign = useCallback(async () => {
    if (!selectedAlignMol || !selectedLigandId || !pdbInfo) return;
    const ligand = pdbInfo.ligands.find((l) => l.ligand_id === selectedLigandId);
    if (!ligand) return;
    setAligning(true);
    setAlignError(null);
    setAlignmentResult(null);
    try {
      const resp = await axios.post<AlignmentResult>(`${API_URL}/align-to-ligand`, {
        mol_smiles: selectedAlignMol,
        ligand_sdf: ligand.sdf,
      });
      setAlignmentResult(resp.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setAlignError(err.response.data.detail);
      } else {
        setAlignError("Alignment failed.");
      }
    } finally {
      setAligning(false);
    }
  }, [selectedAlignMol, selectedLigandId, pdbInfo]);

  const handleClearPdb = useCallback(() => {
    setPdbInfo(null);
    setPdbIdInput("");
    setPdbError(null);
    setAlignmentResult(null);
    setAlignError(null);
    setSelectedLigandId("");
    setSelectedAlignMol("");
  }, []);

  // Build Mol* viewer HTML
  const molstarHtml = useMemo(() => {
    if (!pdbInfo) return "";
    const pdbB64 = btoa(pdbInfo.pdb_content);
    const r = parseInt(bgColor.slice(1, 3), 16);
    const g = parseInt(bgColor.slice(3, 5), 16);
    const b = parseInt(bgColor.slice(5, 7), 16);

    // Use element-symbol theme with carbonColor override on het/ligand components.
    // Accesses the underlying Mol* plugin hierarchy directly — only touches non-polymer,
    // non-water components so only ligand carbons turn cyan while N/O/S keep CPK colors.
    const colorCarbonScript = `
      async function colorHetCarbonsCyan() {
        await new Promise(function(res) { setTimeout(res, 1000); });
        try {
          var plugin = viewerInstance.plugin;
          var structures = plugin.managers.structure.hierarchy.current.structures;
          for (var si = 0; si < structures.length; si++) {
            var comps = structures[si].components;
            for (var ci = 0; ci < comps.length; ci++) {
              var comp = comps[ci];
              var key = (comp.key || '').toLowerCase();
              if (key === 'polymer' || key === 'water' || key === 'coarse') continue;
              await plugin.managers.structure.component.updateRepresentationsTheme(
                [comp],
                {
                  color: 'element-symbol',
                  colorParams: {
                    carbonColor: { name: 'uniform', params: { value: 0x00CCCC } }
                  }
                }
              );
            }
          }
        } catch(e) { console.warn('cyan carbon coloring failed:', e); }
      }
    `;

    const alignedBlock = alignmentResult
      ? colorCarbonScript + `
        var alignedMolLoaded = false;
        viewerInstance.events.loadComplete.subscribe(async function() {
          try { viewerInstance.plugin.canvas3d.setProps({ camera: { mode: 'orthographic' } }); } catch(e) {}
          await viewerInstance.visual.update({ polymer: { type: 'cartoon', colorScheme: 'chain-id' }, het: { type: 'ball-and-stick' }, water: false });
          await colorHetCarbonsCyan();
          if (!alignedMolLoaded) {
            alignedMolLoaded = true;
            viewerInstance.load({ url: 'data:text/plain;base64,${btoa(alignmentResult.aligned_sdf)}', format: 'mol', isBinary: false }, false)
              .then(function() { colorHetCarbonsCyan(); });
          }
        });
      `
      : colorCarbonScript + `
        viewerInstance.events.loadComplete.subscribe(async function() {
          try { viewerInstance.plugin.canvas3d.setProps({ camera: { mode: 'orthographic' } }); } catch(e) {}
          await viewerInstance.visual.update({ polymer: { type: 'cartoon', colorScheme: 'chain-id' }, het: { type: 'ball-and-stick' }, water: false });
          await colorHetCarbonsCyan();
        });
      `;

    return `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://www.ebi.ac.uk/pdbe/pdb-component-library/css/pdbe-molstar-3.1.3.css">
  <script src="https://www.ebi.ac.uk/pdbe/pdb-component-library/js/pdbe-molstar-plugin-3.1.3.js"></script>
  <style>body{margin:0;padding:0}#mv{width:100%;height:550px;position:relative;border-radius:8px;overflow:hidden}</style>
</head>
<body>
  <div id="mv"></div>
  <script>
    var viewerInstance = new PDBeMolstarPlugin();
    viewerInstance.render(document.getElementById('mv'), {
      customData: { url: 'data:text/plain;base64,${pdbB64}', format: 'pdb' },
      alphafoldView: false,
      bgColor: { r: ${r}, g: ${g}, b: ${b} },
      hideControls: false,
      hideCanvasControls: ['selection','animation','expand'],
      sequencePanel: true,
      landscape: true,
      reactive: true
    });
    ${alignedBlock}
  </script>
</body>
</html>`;
  }, [pdbInfo, bgColor, alignmentResult]);

  const fragTypeColor = (t: string) => {
    switch (t) {
      case "ring":
        return "primary";
      case "fused_ring":
        return "secondary";
      case "linker":
        return "warning";
      case "terminal":
        return "info";
      default:
        return "default";
    }
  };

  return (
    <ErrorBoundary fallbackMessage="Something went wrong. Please refresh the page.">
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Title */}
      <Box sx={{ mb: 3, display: "flex", alignItems: "center", gap: 1 }}>
        <Box sx={{ flex: 1 }} />
        <ScienceIcon sx={{ fontSize: 36, color: "primary.main" }} />
        <Typography variant="h4" fontWeight={700}>
          Morpheus V
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Tooltip title={darkMode ? "Switch to light mode" : "Switch to dark mode"}>
          <IconButton onClick={() => setDarkMode((d) => !d)} color="inherit">
            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, textAlign: "center" }}>
        A tool for scaffold hopping and bioisosteric replacement.
      </Typography>

      {/* ---- Input Section ---- */}
      <Paper elevation={1} sx={{ p: 3, mb: 2 }}>
        <Box
          sx={{
            display: "flex",
            gap: 2,
            alignItems: "flex-start",
            flexWrap: "wrap",
          }}
        >
          {/* SMILES text input */}
          <TextField
            label="Enter SMILES string"
            value={smilesInput}
            onChange={(e) => setSmilesInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleFragment();
            }}
            fullWidth
            sx={{
              flex: 3,
              minWidth: 300,
              "& .MuiOutlinedInput-root": { borderRadius: 3 },
            }}
            size="medium"
          />

          {/* Example dropdown */}
          <FormControl size="medium" sx={{ flex: 1, minWidth: 180 }}>
            <InputLabel>Examples</InputLabel>
            <Select
              value={selectedExample}
              label="Examples"
              onChange={(e) => handleExampleChange(e.target.value)}
            >
              <MenuItem value="">
                <em>-- Select an example --</em>
              </MenuItem>
              {Object.keys(EXAMPLE_MOLECULES)
                .filter((k) => k !== "")
                .map((name) => (
                  <MenuItem key={name} value={name}>
                    {name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>

          <FormControl size="medium" sx={{ minWidth: 130 }}>
            <InputLabel>Max terminal atoms</InputLabel>
            <Select
              value={maxTerminalAtoms}
              label="Max terminal atoms"
              onChange={(e) => setMaxTerminalAtoms(Number(e.target.value))}
            >
              <MenuItem value={3}>3 atoms</MenuItem>
              <MenuItem value={4}>4 atoms</MenuItem>
              <MenuItem value={5}>5 atoms</MenuItem>
            </Select>
          </FormControl>
          {loading && <CircularProgress size={28} sx={{ alignSelf: "center" }} />}
        </Box>
      </Paper>

      {/* ---- Ketcher Expander ---- */}
      <Accordion
        expanded={ketcherExpanded}
        onChange={(_, exp) => setKetcherExpanded(exp)}
        sx={{ mb: 3 }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <EditIcon fontSize="small" />
            <Typography fontWeight={600}>Draw Molecule (Ketcher)</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Suspense fallback={<Box sx={{ p: 3, textAlign: "center" }}><CircularProgress /></Box>}>
            <KetcherEditor onSmilesChange={(s) => setSmilesInput(s)} darkMode={darkMode} />
          </Suspense>
        </AccordionDetails>
      </Accordion>

      {/* ---- Error ---- */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* ---- Results ---- */}
      {result && (
        <>
          {/* Input Molecule */}
          <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2, textAlign: "center" }}>
              Input Molecule
            </Typography>
            <Box
              sx={{
                display: "flex",
                gap: 3,
                alignItems: "stretch",
                flexWrap: "nowrap",
                overflowX: "auto",
              }}
            >
              {/* 2D structure */}
              <Box sx={{ textAlign: "center", flex: "0 0 340px" }}>
                <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 0.5 }}>
                  2D Structure
                </Typography>
                <Box
                  sx={{
                    width: 340,
                    height: 340,
                    border: "1px solid #e0e0e0",
                    borderRadius: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden",
                  }}
                >
                  <img
                    src={`data:image/png;base64,${result.molecule.image}`}
                    alt="Input molecule"
                    style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 4 }}
                  />
                </Box>
              </Box>

              {/* 3D structure */}
              {result.molecule.molblock_3d && (
                <Box sx={{ textAlign: "center", flex: "0 0 340px" }}>
                  <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 0.5 }}>
                    3D Structure
                  </Typography>
                  <Box
                    ref={viewerContainerRef}
                    sx={{
                      width: 340,
                      height: 340,
                      border: "1px solid #e0e0e0",
                      borderRadius: 1,
                      position: "relative",
                    }}
                  />
                </Box>
              )}

              {/* Molecule properties */}
              <Box sx={{ flex: 1, minWidth: 200 }}>
                <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 0.5, textAlign: "center" }}>
                  Molecule Properties
                </Typography>
                <Box
                  sx={{
                    height: 340,
                    border: "1px solid #e0e0e0",
                    borderRadius: 1,
                    p: 2,
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                  }}
                >
                  <Table size="small">
                    <TableBody>
                      {[
                        ["Rings", result.molecule.num_rings],
                        ["MW", result.molecule.mw],
                        ["HBD", result.molecule.hbd],
                        ["HBA", result.molecule.hba],
                        ["TPSA", `${result.molecule.tpsa} Å²`],
                        ["cLogP", result.molecule.clogp],
                        ["QED", result.molecule.qed],
                        ["SA Score", result.molecule.sascore ?? "—"],
                      ].map(([label, value]) => (
                        <TableRow key={label as string}>
                          <TableCell sx={{ fontWeight: 600, border: 0, py: 0.8, px: 1 }}>
                            {label}
                          </TableCell>
                          <TableCell sx={{ border: 0, py: 0.8, px: 1 }}>
                            {value}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Box>
              </Box>
            </Box>
          </Paper>

          <Divider sx={{ mb: 3 }} />

          {/* Fragments */}
          <Paper elevation={1} sx={{ p: 3 }}>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 0.5, textAlign: "center" }}>
              Fragments ({result.displayed_fragments} displayed,{" "}
              {result.total_fragments} total)
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2, textAlign: "center" }}>
              Click a fragment to select it.
            </Typography>

            <Grid container spacing={2} justifyContent="center">
              {result.fragments.map((frag, idx) => {
                const isSelected = selectedFragIdx === idx;
                return (
                  <Grid size={{ xs: 6, sm: 4, md: 3, lg: 2 }} key={idx}>
                    <Card
                      onClick={() => setSelectedFragIdx(idx)}
                      sx={{
                        cursor: "pointer",
                        border: isSelected ? "3px solid #1976d2" : "1px solid",
                        borderColor: isSelected ? "#1976d2" : "divider",
                        bgcolor: isSelected
                          ? "rgba(25, 118, 210, 0.08)"
                          : "background.paper",
                        transition: "all 0.15s",
                        "&:hover": {
                          boxShadow: 3,
                          borderColor: "#1976d2",
                        },
                      }}
                    >
                      {isSelected && (
                        <Box
                          sx={{
                            bgcolor: "#1976d2",
                            color: "#fff",
                            textAlign: "center",
                            py: 0.3,
                            fontSize: 12,
                            fontWeight: 700,
                          }}
                        >
                          ✓ Selected
                        </Box>
                      )}
                      <CardMedia
                        component="img"
                        image={`data:image/png;base64,${frag.image}`}
                        alt={frag.wildcard_smiles}
                        sx={{ p: 1 }}
                      />
                      <CardContent
                        sx={{
                          py: 1,
                          px: 1.5,
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          "&:last-child": { pb: 1 },
                        }}
                      >
                        <Chip
                          label={frag.frag_type}
                          size="small"
                          color={fragTypeColor(frag.frag_type) as any}
                          sx={{ mb: 0.5 }}
                        />
                        <Typography
                          variant="caption"
                          component="div"
                          sx={{
                            fontFamily: "monospace",
                            wordBreak: "break-all",
                            fontSize: 11,
                            color: "text.secondary",
                            textAlign: "center",
                          }}
                        >
                          {frag.wildcard_smiles}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>

            {result.fragments.length === 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                No displayable fragments found. The molecule may be too simple
                to decompose.
              </Alert>
            )}

            {result.fragments.length > 0 && selectedFragIdx === null && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Please select a fragment above before searching.
              </Alert>
            )}
          </Paper>

          {/* ---- Search & Replace Section ---- */}
          <Paper elevation={1} sx={{ p: 3, mt: 3 }}>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2, textAlign: "center" }}>
              Search &amp; Replace
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Select a fragment above, adjust parameters, then click{" "}
              <strong>Search &amp; Replace</strong> to find similar fragments
              and generate new molecules.
            </Typography>

            {/* Sliders */}
            <Box sx={{ display: "flex", gap: 4, flexWrap: "wrap", mb: 3 }}>
              <Box sx={{ flex: 1, minWidth: 220 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
                  <Typography variant="body2" fontWeight={600}>
                    Similarity Threshold: {simThreshold.toFixed(2)}
                  </Typography>
                  <Tooltip title="Tanimoto fingerprint similarity cutoff. Only fragments at least this similar to the selected fragment will be retrieved. Higher values → fewer but closer matches." arrow>
                    <InfoOutlinedIcon sx={{ fontSize: 15, color: "text.secondary", cursor: "help" }} />
                  </Tooltip>
                </Box>
                <Slider
                  value={simThreshold}
                  onChange={(_, v) => setSimThreshold(v as number)}
                  min={0.01}
                  max={1.0}
                  step={0.01}
                  valueLabelDisplay="auto"
                  size="small"
                />
              </Box>
              <Box sx={{ flex: 1, minWidth: 220 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
                  <Typography variant="body2" fontWeight={600}>
                    Maximum Results: {maxResults}
                  </Typography>
                  <Tooltip title="Maximum number of similar fragments to retrieve from the selected libraries. Higher values give more candidate molecules but increase search time." arrow>
                    <InfoOutlinedIcon sx={{ fontSize: 15, color: "text.secondary", cursor: "help" }} />
                  </Tooltip>
                </Box>
                <Slider
                  value={maxResults}
                  onChange={(_, v) => setMaxResults(v as number)}
                  min={10}
                  max={500}
                  step={10}
                  valueLabelDisplay="auto"
                  size="small"
                />
              </Box>
            </Box>

            {/* Library selector accordion */}
            <Accordion disableGutters elevation={0} sx={{ mb: 2, border: 1, borderColor: "divider", borderRadius: "8px !important", "&:before": { display: "none" } }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="body2" fontWeight={600}>
                  Fragment Libraries ({selectedLibraries.size}/{fragmentLibraries.length} selected)
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0 }}>
                <Box sx={{ display: "flex", flexWrap: "wrap" }}>
                  {(() => {
                    const ORDER = [
                      "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz",
                      "VeryCommon.txt.gz",
                      "Common.txt.gz",
                      "LessCommon.txt.gz",
                      "Rare.txt.gz",
                      "VeryRare.txt.gz",
                      "ExtremelyRare.txt.gz",
                      "UltraRare.txt.gz",
                      "Doubletons.txt.gz",
                      "Singletons.txt.gz",
                    ];
                    const DISPLAY: Record<string, string> = {
                      "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz": "Entire ChEMBL",
                      "VeryCommon.txt.gz": "Very Common",
                      "Common.txt.gz": "Common",
                      "LessCommon.txt.gz": "Less Common",
                      "Rare.txt.gz": "Rare",
                      "VeryRare.txt.gz": "Very Rare",
                      "ExtremelyRare.txt.gz": "Extremely Rare",
                      "UltraRare.txt.gz": "Ultra Rare",
                      "Doubletons.txt.gz": "Doubletons",
                      "Singletons.txt.gz": "Singletons",
                    };
                    const TIPS: Record<string, string> = {
                      "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz":
                        "Complete ChEMBL fragment library — contains all fragments regardless of frequency. Slowest but most comprehensive.",
                      "VeryCommon.txt.gz":
                        "Fragments appearing in ≥ 725 bioactive ChEMBL molecules. 325 fragments — highest synthesizability & drug-likeness confidence.",
                      "Common.txt.gz":
                        "Fragments appearing in 215–724 bioactive ChEMBL molecules. 750 fragments — high synthetic accessibility & broad drug-like coverage.",
                      "LessCommon.txt.gz":
                        "Fragments appearing in 65–214 bioactive ChEMBL molecules. 1,863 fragments — good balance of diversity and frequency.",
                      "Rare.txt.gz":
                        "Fragments appearing in 25–64 bioactive ChEMBL molecules. 3,633 fragments — includes less-frequent fragments for broader exploration.",
                      "VeryRare.txt.gz":
                        "Fragments appearing in 9–24 bioactive ChEMBL molecules. 8,164 fragments — useful for exploring unusual chemical space.",
                      "ExtremelyRare.txt.gz":
                        "Fragments appearing in 5–8 bioactive ChEMBL molecules. 8,067 fragments — low-frequency fragments for novel analogue exploration.",
                      "UltraRare.txt.gz":
                        "Fragments appearing in 3–4 bioactive ChEMBL molecules. 11,111 fragments — highly unusual fragments; use for maximum diversity.",
                      "Doubletons.txt.gz":
                        "Fragments appearing in exactly 2 bioactive ChEMBL molecules — the rarest validated fragments. Use for maximum structural novelty and scaffold hopping.",
                      "Singletons.txt.gz":
                        "Fragments appearing in exactly 1 bioactive ChEMBL molecule — unique, one-of-a-kind fragments. Highest novelty; proceed with caution as chemical space is largely uncharted.",
                    };
                    return [...fragmentLibraries]
                      .sort((a, b) => {
                        const ia = ORDER.indexOf(a);
                        const ib = ORDER.indexOf(b);
                        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
                      })
                      .map((name) => {
                        const displayName = DISPLAY[name] ?? name.replace(/\.txt\.gz$/, "").replace(/([a-z])([A-Z])/g, "$1 $2");
                        const tip = TIPS[name] ?? name;
                        const CHEMBL = "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz";
                        const chemblSelected = selectedLibraries.has(CHEMBL);
                        const isChembl = name === CHEMBL;
                        const isDisabled = !isChembl && chemblSelected;
                        return (
                          <Tooltip key={name} title={isDisabled ? "Deselect 'Entire ChEMBL' to choose individual libraries" : tip} arrow placement="top">
                            <FormControlLabel
                              control={
                                <Checkbox
                                  size="small"
                                  checked={selectedLibraries.has(name)}
                                  disabled={isDisabled}
                                  onChange={(e) => {
                                    if (isChembl && e.target.checked) {
                                      // Selecting ChEMBL clears all others
                                      setSelectedLibraries(new Set([CHEMBL]));
                                    } else {
                                      const next = new Set(selectedLibraries);
                                      if (e.target.checked) next.add(name);
                                      else next.delete(name);
                                      setSelectedLibraries(next);
                                    }
                                  }}
                                />
                              }
                              label={<Typography variant="body2">{displayName}</Typography>}
                            />
                          </Tooltip>
                        );
                      });
                  })()}
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Search & Replace button */}
            <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
              <Button
                variant="contained"
                startIcon={searching ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                onClick={handleSearchReplace}
                disabled={selectedFragIdx === null || searching || selectedLibraries.size === 0}
                sx={{
                  borderRadius: "50px",
                  px: 5,
                  py: 1.5,
                  fontSize: "1.05rem",
                  fontWeight: 700,
                  minWidth: 240,
                  textTransform: "none",
                  boxShadow: 3,
                }}
              >
                {searching ? "Searching…" : "Find Similar Fragments"}
              </Button>
            </Box>

            {selectedLibraries.size === 0 && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                No fragment libraries selected. Tick at least one library above to enable searching.
              </Alert>
            )}

            {/* Progress bar */}
            {searching && (
              <Box sx={{ mb: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={searchProgress * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                  {searchStage === "searching" && <>Searching fragment library… {Math.round(searchProgress * 100)}%</>}
                  {searchStage === "generating" && <>Generating molecules… {Math.round(searchProgress * 100)}%</>}
                  {searchStage === "scoring" && <>Scoring molecules… {Math.round(searchProgress * 100)}%</>}
                  {searchStage === "done" && <>Finalizing… 100%</>}
                  {!["searching", "generating", "scoring", "done"].includes(searchStage) && <>{Math.round(searchProgress * 100)}%</>}
                </Typography>
              </Box>
            )}

            {searchError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {searchError}
              </Alert>
            )}

            {/* Generated Molecules Grid */}
            {generatedMols.length > 0 && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" fontWeight={600} sx={{ mb: 0.5 }}>
                  Generated Molecules ({showAllMols ? generatedMols.length : filteredMols.length} shown
                  {!showAllMols && filteredMols.length !== generatedMols.length && ` of ${generatedMols.length}`}
                  {` — ${rejectedMols.length} rejected — from ${totalSimilar} similar fragments`})
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Original molecule with selected fragment replaced by each similar fragment.
                </Typography>

                {/* Parallel Coordinates Filter */}
                {Object.keys(molFilters).length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" fontWeight={600} sx={{ mb: 0.5, textAlign: "center" }}>
                      💡 Filter molecules by dragging on the axes below
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                      <Plot
                        data={[{
                          type: "parcoords" as const,
                          line: {
                            color: generatedMols.map((_, i) => i),
                            colorscale: "Viridis",
                          } as any,
                          customdata: generatedMols.map((m) => [
                            m.new_fragment_smiles || "",
                            (m.frag_similarity ?? 0).toFixed(3),
                          ]) as any,
                          hovertemplate:
                            "Fragment: %{customdata[0]}<br>Frag. Sim: %{customdata[1]}<extra></extra>" as any,
                          dimensions: DESCRIPTOR_KEYS
                            .filter((d) => molFilters[d.key] != null)
                            .map(({ key, label }) => {
                              const f = molFilters[key];
                              const dim: any = {
                                label,
                                values: generatedMols.map((m) => (m as any)[key] ?? null),
                                range: [f.min, f.max] as [number, number],
                                constraintrange: [f.currentMin, f.currentMax] as [number, number],
                              };
                              if (key === "frag_library_idx" && libIndexLabels.length > 0) {
                                dim.tickvals = libIndexLabels.map((_, i) => i);
                                dim.ticktext = libIndexLabels;
                              }
                              return dim;
                            }),
                          unselected: { line: { color: "lightgray", opacity: 0.3 } },
                        }] as any}
                        layout={{
                          margin: { t: 60, r: 40, b: 20, l: 40 },
                          height: 320,
                          paper_bgcolor: "transparent",
                          font: { color: darkMode ? "#e0e0e0" : "#333", size: 13 },
                        }}
                        config={{ responsive: true }}
                        style={{
                          width: "100%",
                          minWidth:
                            DESCRIPTOR_KEYS.filter((d) => molFilters[d.key] != null).length > 6
                              ? `${DESCRIPTOR_KEYS.filter((d) => molFilters[d.key] != null).length * 110}px`
                              : "100%",
                        }}
                        onRestyle={(eventData: any) => {
                          if (!eventData?.[0]) return;
                          const updates = eventData[0];
                          const availKeys = DESCRIPTOR_KEYS.filter((d) => molFilters[d.key] != null);
                          Object.keys(updates).forEach((k) => {
                            const match = k.match(/dimensions\[(\d+)\]\.constraintrange/);
                            if (!match) return;
                            const desc = availKeys[parseInt(match[1])];
                            if (!desc) return;
                            const colKey = desc.key;
                            const range = updates[k];
                            if (!range || range.length === 0) {
                              setMolFilters((prev) => ({
                                ...prev,
                                [colKey]: { ...prev[colKey], currentMin: prev[colKey].min, currentMax: prev[colKey].max },
                              }));
                            } else if (Array.isArray(range[0])) {
                              setMolFilters((prev) => ({
                                ...prev,
                                [colKey]: { ...prev[colKey], currentMin: range[0][0], currentMax: range[0][1] },
                              }));
                            } else {
                              setMolFilters((prev) => ({
                                ...prev,
                                [colKey]: { ...prev[colKey], currentMin: range[0], currentMax: range[1] },
                              }));
                            }
                          });
                        }}
                      />
                    </Box>
                    {/* Legend column picker */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
                      <Typography variant="body2" fontWeight={600} sx={{ mr: 0.5 }}>
                        Show as legend:
                      </Typography>
                      {DESCRIPTOR_KEYS.map(({ key, label }) => {
                        const active = legendCols.includes(key);
                        return (
                          <Chip
                            key={key}
                            label={label}
                            size="small"
                            color={active ? "primary" : "default"}
                            variant={active ? "filled" : "outlined"}
                            onClick={() =>
                              setLegendCols((prev) =>
                                active ? prev.filter((c) => c !== key) : [...prev, key]
                              )
                            }
                            sx={{ cursor: "pointer", fontWeight: active ? 700 : 400 }}
                          />
                        );
                      })}
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 2, mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        <strong>{filteredMols.length}</strong> / {generatedMols.length} molecules pass filters
                      </Typography>
                      <Button
                        size="small"
                        variant={showAllMols ? "contained" : "outlined"}
                        onClick={() => { setShowAllMols((s) => !s); setMolPage(1); }}
                        sx={{ borderRadius: "50px", textTransform: "none" }}
                      >
                        {showAllMols ? "Showing all — click to filter" : "Show all (ignore filters)"}
                      </Button>
                    </Box>
                  </Box>
                )}

                {(() => {
                  const baseMols = showAllMols ? generatedMols : filteredMols;
                  const subMols = substructMatchSmiles !== null
                    ? baseMols.filter((m) => substructExclude
                        ? !substructMatchSmiles.has(m.smiles)
                        : substructMatchSmiles.has(m.smiles))
                    : baseMols;
                  const displayMols = [...subMols].sort((a, b) => {
                    const va = (a as any)[sortKey];
                    const vb = (b as any)[sortKey];
                    if (va == null && vb == null) return 0;
                    if (va == null) return 1;
                    if (vb == null) return -1;
                    return sortDir === "asc" ? va - vb : vb - va;
                  });
                  const totalPages = Math.ceil(displayMols.length / MOLS_PER_PAGE);
                  const pagedMols = displayMols.slice((molPage - 1) * MOLS_PER_PAGE, molPage * MOLS_PER_PAGE);
                  return (
                    <>
                      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5, flexWrap: "wrap", gap: 1 }}>
                        {/* Sort controls */}
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                          <Typography variant="body2" fontWeight={600} sx={{ mr: 0.5 }}>
                            Sort by:
                          </Typography>
                          <FormControl size="small" sx={{ minWidth: 160 }}>
                            <Select
                              value={sortKey}
                              onChange={(e) => { setSortKey(e.target.value); setMolPage(1); }}
                              sx={{ fontSize: 13 }}
                            >
                              {DESCRIPTOR_KEYS.map(({ key, label }) => (
                                <MenuItem key={key} value={key}>{label}</MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                          <Tooltip title={sortDir === "asc" ? "Ascending — click to switch to descending" : "Descending — click to switch to ascending"}>
                            <IconButton
                              size="small"
                              onClick={() => { setSortDir((d) => d === "asc" ? "desc" : "asc"); setMolPage(1); }}
                              color="primary"
                            >
                              {sortDir === "asc" ? <ArrowUpwardIcon fontSize="small" /> : <ArrowDownwardIcon fontSize="small" />}
                            </IconButton>
                          </Tooltip>
                          <Typography variant="caption" color="text.secondary">
                            {/* {sortDir === "asc" ? "Ascending" : "Descending"} */}
                          </Typography>
                        </Box>
                        <Box sx={{ display: "flex", alignItems: "center" }}>
                          {/* Substructure exclude toggle */}
                          <Tooltip title={substructExclude ? "Exclude mode: hiding molecules with substructure" : "Include mode: showing only molecules with substructure"}>
                            <Box sx={{ display: "flex", alignItems: "center", mr: 0.5 }}>
                              <Typography variant="caption" color={!substructExclude ? "primary" : "text.secondary"} sx={{ fontWeight: 700, fontSize: 11 }}>IN</Typography>
                              <Switch
                                size="small"
                                checked={substructExclude}
                                onChange={(e) => { setSubstructExclude(e.target.checked); setMolPage(1); }}
                                color="error"
                                sx={{ mx: 0.25 }}
                              />
                              <Typography variant="caption" color={substructExclude ? "error" : "text.secondary"} sx={{ fontWeight: 700, fontSize: 11 }}>EX</Typography>
                            </Box>
                          </Tooltip>
                          {/* Substructure filter input */}
                          <Tooltip title={substructMatchSmiles !== null ? `Filtering by: ${substructInput}` : "Enter SMILES or SMARTS and press Enter to filter"}>
                            <Box
                              component="form"
                              onSubmit={async (e: React.FormEvent) => {
                                e.preventDefault();
                                const pat = substructInput.trim();
                                if (!pat) {
                                  setSubstructMatchSmiles(null);
                                  setSubstructError(null);
                                  return;
                                }
                                setSubstructError(null);
                                try {
                                  const resp = await axios.post<{ matching_smiles: string[] }>(
                                    `${API_URL}/substructure-match`,
                                    { pattern: pat, smiles_list: generatedMols.map((m) => m.smiles) },
                                  );
                                  setSubstructMatchSmiles(new Set(resp.data.matching_smiles));
                                  setMolPage(1);
                                } catch (err: any) {
                                  setSubstructError(err.response?.data?.detail || "Invalid pattern");
                                  setSubstructMatchSmiles(null);
                                }
                              }}
                              sx={{ display: "flex", alignItems: "center" }}
                            >
                              <TextField
                                value={substructInput}
                                onChange={(e) => {
                                  setSubstructInput(e.target.value);
                                  if (!e.target.value.trim()) {
                                    setSubstructMatchSmiles(null);
                                    setSubstructError(null);
                                  }
                                }}
                                placeholder="Substructure filter…"
                                size="small"
                                error={!!substructError}
                                title={substructError || ""}
                                slotProps={{
                                  input: {
                                    endAdornment: substructMatchSmiles !== null ? (
                                      <IconButton
                                        size="small"
                                        onClick={() => { setSubstructInput(""); setSubstructMatchSmiles(null); setSubstructError(null); }}
                                        sx={{ p: 0.3 }}
                                      >
                                        <ClearIcon sx={{ fontSize: 14 }} />
                                      </IconButton>
                                    ) : null,
                                  },
                                }}
                                sx={{
                                  mr: 1,
                                  "& .MuiOutlinedInput-root": {
                                    borderRadius: "50px",
                                    fontSize: 13,
                                    bgcolor: substructMatchSmiles !== null
                                      ? "rgba(25,118,210,0.08)"
                                      : "background.paper",
                                    "& fieldset": {
                                      borderColor: substructMatchSmiles !== null ? "primary.main" : undefined,
                                    },
                                  },
                                  width: 210,
                                }}
                              />
                            </Box>
                          </Tooltip>
                          <Pagination
                            count={totalPages}
                            page={molPage}
                            onChange={(_e, p) => setMolPage(p)}
                            color="primary"
                            showFirstButton
                            showLastButton
                          />
                          <Tooltip title="More options">
                            <IconButton size="small" onClick={(e) => setMolGridMenuAnchor(e.currentTarget)} sx={{ ml: 0.5 }}>
                              <MoreVertIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </Box>
                      <Menu
                        anchorEl={molGridMenuAnchor}
                        open={Boolean(molGridMenuAnchor)}
                        onClose={() => setMolGridMenuAnchor(null)}
                        transformOrigin={{ horizontal: "right", vertical: "top" }}
                        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
                      >
                        <MenuItem
                          onClick={() => {
                            setSelectedMolIds(new Set(displayMols.map((m) => m.id)));
                            setMolGridMenuAnchor(null);
                          }}
                        >
                          Select all
                        </MenuItem>
                        <MenuItem
                          onClick={() => {
                            setSelectedMolIds(new Set());
                            setMolGridMenuAnchor(null);
                          }}
                        >
                          Unselect all
                        </MenuItem>
                        <MenuItem
                          onClick={() => {
                            const mols = selectedMolIds.size > 0
                              ? displayMols.filter((m) => selectedMolIds.has(m.id))
                              : displayMols;
                            const headers = [
                              "id", "smiles", "new_fragment_smiles", "frag_similarity",
                              "mol_similarity", "mw", "clogp", "qed", "tpsa",
                              "hbd", "hba", "n_aliphatic_rings", "n_aromatic_rings",
                              "n_rotatable_bonds", "sascore", "mscore",
                            ];
                            const csvRows = mols.map((m) =>
                              headers.map((h) => {
                                const v = (m as any)[h];
                                if (v == null) return "";
                                const s = String(v);
                                return s.includes(",") || s.includes('"') || s.includes("\n")
                                  ? `"${s.replace(/"/g, '""')}"` : s;
                              }).join(",")
                            );
                            navigator.clipboard.writeText([headers.join(","), ...csvRows].join("\n"));
                            setMolGridMenuAnchor(null);
                          }}
                        >
                          Copy to clipboard
                        </MenuItem>
                        <Divider />
                        <MenuItem
                          onClick={() => {
                            const mols = selectedMolIds.size > 0
                              ? displayMols.filter((m) => selectedMolIds.has(m.id))
                              : displayMols;
                            const lines = mols.map((m) => m.smiles).join("\n");
                            const blob = new Blob([lines], { type: "chemical/x-daylight-smiles" });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url; a.download = "molecules.smi"; a.click();
                            URL.revokeObjectURL(url);
                            setMolGridMenuAnchor(null);
                          }}
                        >
                          Save SMILES
                        </MenuItem>
                        <MenuItem
                          onClick={() => {
                            const mols = selectedMolIds.size > 0
                              ? displayMols.filter((m) => selectedMolIds.has(m.id))
                              : displayMols;
                            const headers = [
                              "id", "smiles", "new_fragment_smiles", "frag_similarity",
                              "mol_similarity", "mw", "clogp", "qed", "tpsa",
                              "hbd", "hba", "n_aliphatic_rings", "n_aromatic_rings",
                              "n_rotatable_bonds", "sascore", "mscore",
                            ];
                            const csvRows = mols.map((m) =>
                              headers.map((h) => {
                                const v = (m as any)[h];
                                if (v == null) return "";
                                const s = String(v);
                                return s.includes(",") || s.includes('"') || s.includes("\n")
                                  ? `"${s.replace(/"/g, '""')}"` : s;
                              }).join(",")
                            );
                            const blob = new Blob([[headers.join(","), ...csvRows].join("\n")], { type: "text/csv" });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url; a.download = "molecules.csv"; a.click();
                            URL.revokeObjectURL(url);
                            setMolGridMenuAnchor(null);
                          }}
                        >
                          Save as CSV
                        </MenuItem>
                        <MenuItem
                          onClick={async () => {
                            const mols = selectedMolIds.size > 0
                              ? displayMols.filter((m) => selectedMolIds.has(m.id))
                              : displayMols;
                            try {
                              const resp = await axios.post(
                                `${API_URL}/export-sdf`,
                                { molecules: mols },
                                { responseType: "blob" },
                              );
                              const url = URL.createObjectURL(resp.data);
                              const a = document.createElement("a");
                              a.href = url; a.download = "molecules.sdf"; a.click();
                              URL.revokeObjectURL(url);
                            } catch { /* silently fail */ }
                            setMolGridMenuAnchor(null);
                          }}
                        >
                          Save as SDF
                        </MenuItem>
                      </Menu>

                <Grid container spacing={2} columns={4}>
                  {pagedMols.map((mol, idx) => (
                    <Grid size={1} key={(molPage - 1) * MOLS_PER_PAGE + idx}>
                      <Tooltip
                        title={
                          <Box sx={{ p: 0.5 }}>
                            {[
                              ["MW", mol.mw],
                              ["cLogP", mol.clogp],
                              ["TPSA", mol.tpsa != null ? `${mol.tpsa} Å²` : null],
                              ["HBD", mol.hbd],
                              ["HBA", mol.hba],
                              ["Ali. Rings", mol.n_aliphatic_rings],
                              ["Aro. Rings", mol.n_aromatic_rings],
                              ["Rot. Bonds", mol.n_rotatable_bonds],
                              ["SA Score", mol.sascore],
                              ["Library", mol.frag_library
                                ? (({
                                    "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz": "ChEMBL",
                                    "VeryCommon.txt.gz": "Very Common",
                                    "Common.txt.gz": "Common",
                                    "LessCommon.txt.gz": "Less Common",
                                    "Rare.txt.gz": "Rare",
                                    "VeryRare.txt.gz": "Very Rare",
                                    "ExtremelyRare.txt.gz": "Ext. Rare",
                                    "UltraRare.txt.gz": "Ultra Rare",
                                    "Doubletons.txt.gz": "Doubletons",
                                    "Singletons.txt.gz": "Singletons",
                                  } as Record<string, string>)[mol.frag_library] ?? mol.frag_library.replace(/\.txt\.gz$/, ""))
                                : null],
                            ].map(([label, val]) => val != null && (
                              <Box key={label as string} sx={{ display: "flex", gap: 1 }}>
                                <Typography variant="caption" sx={{ fontWeight: 700, minWidth: 70 }}>{label}</Typography>
                                <Typography variant="caption">{val}</Typography>
                              </Box>
                            ))}
                          </Box>
                        }
                        placement="top"
                        arrow
                      >
                      <Card
                        onClick={() =>
                          setSelectedMolIds((prev) => {
                            const next = new Set(prev);
                            if (next.has(mol.id)) next.delete(mol.id);
                            else next.add(mol.id);
                            return next;
                          })
                        }
                        sx={{
                          border: selectedMolIds.has(mol.id) ? "2px solid" : "1px solid",
                          borderColor: selectedMolIds.has(mol.id) ? "#42a5f5" : "divider",
                          bgcolor: selectedMolIds.has(mol.id)
                            ? "rgba(66,165,245,0.08)"
                            : "background.paper",
                          transition: "all 0.15s",
                          cursor: "pointer",
                          position: "relative",
                          "&:hover": { boxShadow: 3 },
                        }}
                      >
                        {selectedMolIds.has(mol.id) && (
                          <Box
                            sx={{
                              position: "absolute",
                              top: 6,
                              right: 6,
                              zIndex: 2,
                              bgcolor: "#42a5f5",
                              borderRadius: "50%",
                              width: 22,
                              height: 22,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              boxShadow: 1,
                            }}
                          >
                            <CheckIcon sx={{ fontSize: 14, color: "#fff" }} />
                          </Box>
                        )}
                        <img
                          src={`data:image/png;base64,${mol.image}`}
                          alt=""
                          style={{
                            width: "100%",
                            aspectRatio: "1 / 1",
                            objectFit: "contain",
                            padding: 8,
                            display: "block",
                          }}
                        />
                        <CardContent
                          sx={{
                            py: 0.8,
                            px: 1.5,
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            "&:last-child": { pb: 0.8 },
                          }}
                        >
                          <Box sx={{ display: "flex", alignItems: "center", gap: 0.3, mb: 0.3 }}>
                            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: 11, color: "text.secondary" }}>
                              #{mol.id}
                            </Typography>
                            <Tooltip title={copiedId === mol.id ? "Copied!" : "Copy SMILES"} placement="top">
                              <IconButton
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  navigator.clipboard.writeText(mol.smiles).then(() => {
                                    setCopiedId(mol.id);
                                    setTimeout(() => setCopiedId(null), 1500);
                                  });
                                }}
                                sx={{ p: 0.2, opacity: 0.6, "&:hover": { opacity: 1 } }}
                              >
                                {copiedId === mol.id
                                  ? <CheckIcon sx={{ fontSize: 12, color: "success.main" }} />
                                  : <ContentCopyIcon sx={{ fontSize: 12 }} />}
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", justifyContent: "center" }}>
                            {legendCols.map((key) => {
                              const desc = DESCRIPTOR_KEYS.find((d) => d.key === key);
                              const val = (mol as any)[key];
                              if (val == null || !desc) return null;
                              const display =
                                key === "mol_similarity"
                                  ? `Sim ${(val * 100).toFixed(0)}%`
                                  : key === "shape_sim"
                                  ? `Shape ${(val * 100).toFixed(0)}%`
                                  : key === "esp_sim"
                                  ? `ESP ${(val * 100).toFixed(0)}%`
                                  : key === "shape_esp"
                                  ? `Sh×ESP ${(val * 100).toFixed(0)}%`
                                  : key === "frag_library_idx"
                                  ? `Lib: ${libIndexLabels[val as number] ?? val}`
                                  : `${desc.label} ${val}`;
                              return (
                                <Chip
                                  key={key}
                                  label={display}
                                  size="small"
                                  color={key === "mscore" ? "primary" : "default"}
                                  variant={key === "mscore" ? "filled" : "outlined"}
                                />
                              );
                            })}
                          </Box>
                        </CardContent>
                      </Card>
                      </Tooltip>
                    </Grid>
                  ))}
                </Grid>

                      {/* ---- Similar Fragments accordion ---- */}
                      {similarFragments.length > 0 && (
                        <Accordion disableGutters elevation={0} defaultExpanded={false}
                          sx={{ mt: 2, border: 1, borderColor: "divider", borderRadius: "8px !important", "&:before": { display: "none" } }}>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="body2" fontWeight={600}>
                              Similar Fragments Found ({similarFragments.length})
                            </Typography>
                          </AccordionSummary>
                          <AccordionDetails sx={{ pt: 1 }}>
                            {(() => {
                              // Map fragment SMILES → generated mol ID (or "rejected")
                              const fragToMolId: Record<string, number> = {};
                              generatedMols.forEach((m) => {
                                if (m.new_fragment_smiles) fragToMolId[m.new_fragment_smiles] = m.id;
                              });
                              const rejectedSmiles = new Set(
                                rejectedMols.map((m) => m.new_fragment_smiles).filter(Boolean)
                              );
                              return (
                            <Grid container spacing={1.5} columns={6}>
                              {similarFragments.map((frag, i) => {
                                const libDisplay =
                                  frag.library === "fragments_cleaned_whole_filtered_chembl_with_smiles.txt.gz"
                                    ? "ChEMBL"
                                    : frag.library.replace(/\.txt\.gz$/, "").replace(/([a-z])([A-Z])/g, "$1 $2");
                                const molId = fragToMolId[frag.smiles];
                                const isRejected = rejectedSmiles.has(frag.smiles);
                                return (
                                  <Grid size={1} key={i}>
                                    <Card variant="outlined" sx={{ height: "100%" }}>
                                      <CardMedia
                                        component="img"
                                        image={`data:image/png;base64,${frag.image}`}
                                        alt={frag.smiles}
                                        sx={{ p: 0.5, imageRendering: "-webkit-optimize-contrast" }}
                                      />
                                      <CardContent sx={{ py: 0.5, px: 1, textAlign: "center", "&:last-child": { pb: 0.8 } }}>
                                        {molId != null ? (
                                          <Typography variant="caption" display="block" sx={{ fontWeight: 700, color: "primary.main", mb: 0.2 }}>
                                            Mol #{molId}
                                          </Typography>
                                        ) : isRejected ? (
                                          <Typography variant="caption" display="block" sx={{ fontWeight: 700, color: "error.main", mb: 0.2 }}>
                                            Rejected
                                          </Typography>
                                        ) : (
                                          <Typography variant="caption" display="block" sx={{ fontWeight: 700, color: "text.disabled", mb: 0.2 }}>
                                            No molecule
                                          </Typography>
                                        )}
                                        <Typography variant="caption" display="block" sx={{ fontWeight: 700, color: "text.secondary", mb: 0.2 }}>
                                          Tanimoto Sim: {(frag.similarity * 100).toFixed(1)}%
                                        </Typography>
                                        <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 0.2 }}>
                                          Library: {libDisplay}
                                        </Typography>
                                        <Tooltip title={frag.smiles} placement="bottom" arrow>
                                          <Typography variant="caption" display="block" sx={{
                                            fontSize: 9.5,
                                            color: "text.secondary",
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                            whiteSpace: "nowrap",
                                            cursor: "default",
                                          }}>
                                            {frag.smiles}
                                          </Typography>
                                        </Tooltip>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                );
                              })}
                            </Grid>
                              );
                            })()}
                          </AccordionDetails>
                        </Accordion>
                      )}

                      <Box sx={{ display: "flex", justifyContent: "center", mt: 1.5 }}>
                        <Pagination
                          count={totalPages}
                          page={molPage}
                          onChange={(_e, p) => setMolPage(p)}
                          color="primary"
                          showFirstButton
                          showLastButton
                        />
                      </Box>
                    </>
                  );
                })()}

              </>
            )}
          </Paper>
        </>
      )}

        {/* ---- Protein-Ligand Alignment ---- */}
        {generatedMols.length > 0 && (
          <Accordion sx={{ mt: 3 }} defaultExpanded={false}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, width: "100%" }}>
                <BiotechIcon fontSize="small" color="primary" />
                <Typography fontWeight={600}>Protein-Ligand Alignment</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {/* PDB Input Row */}
              <Box sx={{ display: "flex", gap: 2, alignItems: "flex-end", flexWrap: "wrap", mb: 2, justifyContent: "center" }}>
                {/* Upload PDB file */}
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<CloudUploadIcon />}
                  disabled={fetchingPdb}
                  sx={{ textTransform: "none" }}
                >
                  Upload PDB
                  <input
                    type="file"
                    hidden
                    accept=".pdb,.ent"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleUploadPdb(f);
                      e.target.value = "";
                    }}
                  />
                </Button>

                <Typography variant="body2" color="text.secondary" sx={{ alignSelf: "center" }}>
                  or
                </Typography>

                {/* Fetch by PDB ID */}
                <TextField
                  label="PDB ID"
                  value={pdbIdInput}
                  onChange={(e) => setPdbIdInput(e.target.value.toUpperCase())}
                  size="small"
                  sx={{ width: 120 }}
                  placeholder="e.g. 1ATP"
                  slotProps={{ htmlInput: { maxLength: 4 } }}
                  onKeyDown={(e) => { if (e.key === "Enter") handleFetchPdb(); }}
                />
                <Button
                  variant="contained"
                  onClick={handleFetchPdb}
                  disabled={fetchingPdb || pdbIdInput.trim().length !== 4}
                  size="small"
                  sx={{ textTransform: "none" }}
                >
                  {fetchingPdb ? <CircularProgress size={18} /> : "Fetch"}
                </Button>

                {pdbInfo && (
                  <Button
                    variant="text"
                    color="error"
                    startIcon={<ClearIcon />}
                    onClick={handleClearPdb}
                    size="small"
                    sx={{ textTransform: "none" }}
                  >
                    Clear PDB
                  </Button>
                )}
              </Box>

              {pdbError && <Alert severity="error" sx={{ mb: 2 }}>{pdbError}</Alert>}

              {fetchingPdb && (
                <Box sx={{ display: "flex", justifyContent: "center", py: 3 }}>
                  <CircularProgress />
                </Box>
              )}

              {pdbInfo && !fetchingPdb && (
                <>
                  {/* Structure Info */}
                  <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                    <Typography variant="subtitle2" fontWeight={700} gutterBottom>
                      Structure Info — {pdbInfo.pdb_id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Title:</strong> {pdbInfo.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Atoms:</strong> {pdbInfo.atom_count.toLocaleString()} (incl. {pdbInfo.hetatm_count.toLocaleString()} heteroatoms)
                      {" | "}
                      <strong>Chains:</strong> {pdbInfo.chains.length > 0 ? pdbInfo.chains.join(", ") : "N/A"}
                    </Typography>
                  </Paper>

                  {/* Ligand Cards */}
                  {pdbInfo.ligands.length > 0 ? (
                    <>
                      <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 , textAlign: "center" }}>
                        Extracted Ligands ({pdbInfo.ligands.length})
                      </Typography>
                      <Box
                        sx={{
                          display: "flex",
                          flexWrap: "nowrap",
                          gap: 1.5,
                          justifyContent: "safe center",
                          overflowX: "auto",
                          pb: 1,
                          mb: 2,
                          "&::-webkit-scrollbar": { height: 6 },
                          "&::-webkit-scrollbar-thumb": { bgcolor: "divider", borderRadius: 3 },
                        }}
                      >
                        {pdbInfo.ligands.map((lig) => (
                          <Card
                            key={lig.ligand_id}
                            sx={{
                              width: 160,
                              flexShrink: 0,
                              border: selectedLigandId === lig.ligand_id ? "2px solid" : "1px solid",
                              borderColor: selectedLigandId === lig.ligand_id ? "primary.main" : "divider",
                              cursor: "pointer",
                              transition: "border-color 0.15s",
                              "&:hover": { borderColor: "primary.light" },
                            }}
                            onClick={() => setSelectedLigandId(lig.ligand_id)}
                          >
                            {lig.image && (
                              <CardMedia
                                component="img"
                                image={`data:image/png;base64,${lig.image}`}
                                alt={lig.res_name}
                                sx={{ p: 0.5, bgcolor: "background.default", objectFit: "contain" }}
                              />
                            )}
                            <CardContent sx={{ p: 1, "&:last-child": { pb: 1 } }}>
                              <Typography variant="caption" fontWeight={700}>
                                {lig.res_name} ({lig.chain})
                              </Typography>
                              <Typography variant="caption" display="block" color="text.secondary" sx={{ fontSize: 10 }}>
                                {lig.num_atoms} atoms
                              </Typography>
                              <Typography
                                variant="caption"
                                display="block"
                                sx={{ fontFamily: "monospace", fontSize: 9, wordBreak: "break-all", color: "text.secondary" }}
                              >
                                {lig.smiles.length > 60 ? lig.smiles.slice(0, 57) + "..." : lig.smiles}
                              </Typography>
                              <Tooltip title="Download SDF">
                                <IconButton
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    const blob = new Blob([lig.sdf], { type: "chemical/x-mdl-sdfile" });
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.href = url;
                                    a.download = `${lig.res_name}_${lig.chain}.sdf`;
                                    a.click();
                                    URL.revokeObjectURL(url);
                                  }}
                                  sx={{ mt: 0.5 }}
                                >
                                  <DownloadIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </CardContent>
                          </Card>
                        ))}
                      </Box>
                    </>
                  ) : (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      No ligands found in this PDB structure.
                    </Alert>
                  )}

                  {/* Alignment Controls */}
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1.5, textAlign: "center" }}>
                    Align Generated Molecule → Ligand
                  </Typography>
                  <Box sx={{ display: "flex", gap: 2, alignItems: "flex-end", flexWrap: "wrap", mb: 2, justifyContent: "center" }}>
                    <FormControl size="small" sx={{ minWidth: 220 }}>
                      <InputLabel>Generated Molecule</InputLabel>
                      <Select
                        value={selectedAlignMol}
                        label="Generated Molecule"
                        onChange={(e) => setSelectedAlignMol(e.target.value)}
                      >
                        {generatedMols.slice(0, 50).map((m) => (
                          <MenuItem key={m.id} value={m.smiles}>
                            #{m.id} — MScore {m.mscore ?? "N/A"}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <FormControl size="small" sx={{ minWidth: 180 }}>
                      <InputLabel>Target Ligand</InputLabel>
                      <Select
                        value={selectedLigandId}
                        label="Target Ligand"
                        onChange={(e) => setSelectedLigandId(e.target.value)}
                      >
                        {(pdbInfo?.ligands ?? []).map((l) => (
                          <MenuItem key={l.ligand_id} value={l.ligand_id}>
                            {l.res_name} ({l.chain}) — {l.num_atoms} atoms
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <Button
                      variant="contained"
                      onClick={handleAlign}
                      disabled={aligning || !selectedAlignMol || !selectedLigandId}
                      sx={{ textTransform: "none" }}
                    >
                      {aligning ? <CircularProgress size={18} sx={{ mr: 1 }} /> : null}
                      Align
                    </Button>

                    {alignmentResult && (
                      <Button
                        variant="text"
                        color="error"
                        size="small"
                        onClick={() => { setAlignmentResult(null); setAlignError(null); }}
                        sx={{ textTransform: "none" }}
                      >
                        Clear Alignment
                      </Button>
                    )}
                  </Box>

                  {alignError && <Alert severity="error" sx={{ mb: 2 }}>{alignError}</Alert>}

                  {alignmentResult && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                      RMSD: <strong>{alignmentResult.rmsd != null ? `${alignmentResult.rmsd} Å` : "N/A"}</strong>
                      {" | MCS atoms: "}<strong>{alignmentResult.mcs_atoms}</strong>
                      {" | Conformers sampled: "}<strong>{alignmentResult.num_conformers}</strong>
                    </Alert>
                  )}

                  {/* Background color picker */}
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Viewer background:
                    </Typography>
                    <input
                      type="color"
                      value={bgColor}
                      onChange={(e) => setBgColor(e.target.value)}
                      style={{ width: 28, height: 22, border: "none", cursor: "pointer", background: "none" }}
                    />
                  </Box>

                  {/* Mol* Viewer */}
                  <Box sx={{ borderRadius: 2, overflow: "hidden", border: "1px solid", borderColor: "divider" }}>
                    <iframe
                      ref={molstarIframeRef}
                      srcDoc={molstarHtml}
                      title="PDBe Molstar Viewer"
                      style={{ width: "100%", height: 560, border: "none" }}
                      sandbox="allow-scripts allow-same-origin"
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                    Use mouse to rotate (left-click), zoom (scroll), and pan (right-click). The control panel offers additional visualization options.
                  </Typography>
                </>
              )}
            </AccordionDetails>
          </Accordion>
        )}

        {/* ---- Retrosynthetic Planning ---- */}
        {generatedMols.length > 0 && (
          <Accordion sx={{ mt: 3 }} defaultExpanded={false}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, width: "100%" }}>
                <ScienceIcon fontSize="small" color="primary" />
                <Typography fontWeight={600}>Retrosynthetic Planning</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Select a generated molecule and configure parameters to predict retrosynthetic routes using MCTS tree search.
                The planner service will start automatically on the first request.
              </Typography>

              {/* Molecule selector */}
              <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "flex-end", mb: 2, justifyContent: "center" }}>
                <FormControl size="small" sx={{ minWidth: 280 }}>
                  <InputLabel>Select Molecule</InputLabel>
                  <Select
                    value={retroSelectedSmiles}
                    label="Select Molecule"
                    onChange={(e) => { setRetroSelectedSmiles(e.target.value); setRetroResult(null); setRetroError(null); }}
                  >
                    {generatedMols.slice(0, 50).map((m) => (
                      <MenuItem key={m.id} value={m.smiles}>
                        #{m.id} — MScore {m.mscore ?? "N/A"} — Sim {m.mol_similarity != null ? m.mol_similarity.toFixed(3) : "N/A"}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              {retroSelectedSmiles && (
                <>
                  {/* Selected molecule SMILES */}
                  <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 11, wordBreak: "break-all", mb: 2, textAlign: "center", color: "text.secondary" }}>
                    {retroSelectedSmiles}
                  </Typography>

                  {/* Parameter sliders */}
                  <Box sx={{ display: "flex", gap: 4, flexWrap: "wrap", mb: 3 }}>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="body2" fontWeight={600} gutterBottom>
                        Max Reaction Steps: {retroMaxDepth}
                      </Typography>
                      <Slider value={retroMaxDepth} onChange={(_, v) => setRetroMaxDepth(v as number)} min={1} max={12} step={1} valueLabelDisplay="auto" size="small" />
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="body2" fontWeight={600} gutterBottom>
                        Max MCTS Iterations: {retroMaxIter}
                      </Typography>
                      <Slider value={retroMaxIter} onChange={(_, v) => setRetroMaxIter(v as number)} min={50} max={500} step={50} valueLabelDisplay="auto" size="small" />
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="body2" fontWeight={600} gutterBottom>
                        Min Precursor Size: {retroMinMolSize}
                      </Typography>
                      <Slider value={retroMinMolSize} onChange={(_, v) => setRetroMinMolSize(v as number)} min={1} max={10} step={1} valueLabelDisplay="auto" size="small" />
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="body2" fontWeight={600} gutterBottom>
                        Number of Routes: {retroNumRoutes}
                      </Typography>
                      <Slider value={retroNumRoutes} onChange={(_, v) => setRetroNumRoutes(v as number)} min={1} max={10} step={1} valueLabelDisplay="auto" size="small" />
                    </Box>
                  </Box>

                  {/* Run button */}
                  <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
                    <Button
                      variant="contained"
                      disabled={retroRunning || !retroSelectedSmiles}
                      sx={{ borderRadius: "50px", px: 5, py: 1.5, fontSize: "1.05rem", fontWeight: 700, minWidth: 280, textTransform: "none", boxShadow: 3 }}
                      onClick={async () => {
                        setRetroRunning(true);
                        setRetroResult(null);
                        setRetroError(null);
                        try {
                          const resp = await axios.post(`${API_URL}/plan-synthesis`, {
                            smiles: retroSelectedSmiles,
                            max_routes: retroNumRoutes,
                            max_depth: retroMaxDepth,
                            max_iterations: retroMaxIter,
                            min_mol_size: retroMinMolSize,
                          }, { timeout: 660000 });
                          setRetroResult(resp.data);
                        } catch (err: unknown) {
                          if (axios.isAxiosError(err) && err.response?.data?.detail) {
                            setRetroError(err.response.data.detail);
                          } else {
                            setRetroError("Retrosynthesis request timed out or failed. The first run can take several minutes for initialization. Please try again.");
                          }
                        } finally {
                          setRetroRunning(false);
                        }
                      }}
                    >
                      {retroRunning ? <CircularProgress size={22} color="inherit" sx={{ mr: 1 }} /> : null}
                      {retroRunning ? "Running MCTS…" : "Run Retrosynthetic Planning"}
                    </Button>
                  </Box>

                  {/* Progress */}
                  {retroRunning && (
                    <Box sx={{ mb: 2 }}>
                      <LinearProgress variant="indeterminate" sx={{ height: 6, borderRadius: 3 }} />
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                        Running retrosynthetic planning… The first run may take several minutes while the planner loads building blocks and models.
                      </Typography>
                    </Box>
                  )}

                  {retroError && <Alert severity="error" sx={{ mb: 2 }}>{retroError}</Alert>}

                  {/* Results */}
                  {retroResult && (
                    <>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="h6" fontWeight={600} sx={{ mb: 1, textAlign: "center" }}>
                        Retrosynthetic Routes
                      </Typography>

                      {retroResult.success && retroResult.solved && retroResult.routes.length > 0 ? (
                        <>
                          <Alert severity="success" sx={{ mb: 2 }}>
                            Found {retroResult.routes.length} synthesis route(s)
                          </Alert>
                          <Alert severity="info" sx={{ mb: 2 }}>
                            <strong>Route Score:</strong> Higher scores indicate more favorable routes. Routes are sorted by number of steps (shorter first), then by score.
                          </Alert>

                          {retroResult.routes.map((route, ri) => (
                            <Accordion key={ri} defaultExpanded={ri === 0} sx={{ mb: 1 }}>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Typography fontWeight={600} variant="body2">
                                  Route {ri + 1} — {route.num_steps} step{route.num_steps !== 1 ? "s" : ""} (Score: {route.score.toFixed(4)})
                                </Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                {route.svg ? (
                                  <Box sx={{ bgcolor: "#fff", p: 2, borderRadius: 2, overflowX: "auto", mb: 1, border: "1px solid", borderColor: "divider" }}>
                                    <div dangerouslySetInnerHTML={{ __html: route.svg }} />
                                  </Box>
                                ) : (
                                  <Alert severity="warning" sx={{ mb: 1 }}>
                                    SVG visualization not available{route.svg_error ? `: ${route.svg_error}` : ""}
                                  </Alert>
                                )}

                                {/* Building blocks */}
                                {route.building_blocks.filter((bb) => bb.id).length > 0 && (
                                  <>
                                    <Typography variant="body2" fontWeight={600} sx={{ mt: 1, mb: 0.5 }}>
                                      Building Blocks
                                    </Typography>
                                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                                      {route.building_blocks.filter((bb) => bb.id).map((bb, bi) => (
                                        <Chip
                                          key={bi}
                                          label={`${bb.id}: ${bb.smiles.length > 30 ? bb.smiles.slice(0, 27) + "..." : bb.smiles}`}
                                          size="small"
                                          color="success"
                                          variant="outlined"
                                          onClick={() => {
                                            navigator.clipboard.writeText(bb.smiles);
                                            setRetroCopiedSmiles(bb.smiles);
                                            setTimeout(() => setRetroCopiedSmiles(null), 1500);
                                          }}
                                          icon={retroCopiedSmiles === bb.smiles ? <CheckIcon /> : <ContentCopyIcon />}
                                          sx={{ fontFamily: "monospace", fontSize: 11, cursor: "pointer" }}
                                        />
                                      ))}
                                    </Box>
                                  </>
                                )}

                                {/* Intermediates */}
                                {route.intermediates.length > 0 && (
                                  <>
                                    <Typography variant="body2" fontWeight={600} sx={{ mt: 1, mb: 0.5 }}>
                                      Intermediates
                                    </Typography>
                                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                                      {route.intermediates.map((inter, ii) => (
                                        <Chip
                                          key={ii}
                                          label={`Int-${ii + 1}: ${inter.smiles.length > 35 ? inter.smiles.slice(0, 32) + "..." : inter.smiles}`}
                                          size="small"
                                          color="warning"
                                          variant="outlined"
                                          onClick={() => {
                                            navigator.clipboard.writeText(inter.smiles);
                                            setRetroCopiedSmiles(inter.smiles);
                                            setTimeout(() => setRetroCopiedSmiles(null), 1500);
                                          }}
                                          icon={retroCopiedSmiles === inter.smiles ? <CheckIcon /> : <ContentCopyIcon />}
                                          sx={{ fontFamily: "monospace", fontSize: 11, cursor: "pointer" }}
                                        />
                                      ))}
                                    </Box>
                                  </>
                                )}
                              </AccordionDetails>
                            </Accordion>
                          ))}
                        </>
                      ) : retroResult.success && !retroResult.solved ? (
                        <Alert severity="warning" sx={{ mb: 2 }}>
                          No synthesis route found. Try increasing the maximum reaction steps and/or the number of MCTS iterations. The molecule may be too complex or contain unusual substructures.
                        </Alert>
                      ) : (
                        <Alert severity="error" sx={{ mb: 2 }}>
                          {retroResult.error || "Unknown error"}
                        </Alert>
                      )}

                      {/* Clear results button */}
                      <Box sx={{ display: "flex", justifyContent: "center", mt: 1 }}>
                        <Button
                          variant="text"
                          color="error"
                          size="small"
                          startIcon={<ClearIcon />}
                          onClick={() => { setRetroResult(null); setRetroError(null); }}
                          sx={{ textTransform: "none" }}
                        >
                          Clear Results
                        </Button>
                      </Box>
                    </>
                  )}
                </>
              )}
            </AccordionDetails>
          </Accordion>
        )}

        {/* ---- Rejected Molecules (bottom of app) ---- */}
        {rejectedMols.length > 0 && (
          <Accordion sx={{ mt: 3 }} defaultExpanded={false}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <WarningAmberIcon fontSize="small" color="error" />
                <Typography fontWeight={600} color="error">
                  Rejected Molecules ({rejectedMols.length}) — Structural Alerts Detected
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                These molecules were filtered out because they contain one or more undesirable substructures.
              </Typography>
              <Grid container spacing={2} columns={4}>
                {rejectedMols.map((mol, idx) => (
                  <Grid size={1} key={idx}>
                    <Card
                      sx={{
                        border: "1px solid",
                        borderColor: "error.main",
                        bgcolor: darkMode ? "rgba(211,47,47,0.08)" : "#fff5f5",
                      }}
                    >
                      <CardMedia
                        component="img"
                        image={`data:image/png;base64,${mol.image}`}
                        alt={mol.smiles}
                        sx={{ p: 1 }}
                      />
                      <CardContent
                        sx={{
                          py: 1, px: 1.5,
                          display: "flex", flexDirection: "column", alignItems: "center",
                          "&:last-child": { pb: 1 },
                        }}
                      >
                        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, justifyContent: "center", mb: 0.5 }}>
                          {mol.rejection_reasons.map((r, ri) => (
                            <Chip key={ri} label={r} size="small" color="error" variant="outlined" />
                          ))}
                        </Box>
                        <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", justifyContent: "center", mb: 0.5 }}>
                          {mol.mw != null && <Chip label={`MW ${mol.mw}`} size="small" variant="outlined" />}
                          {mol.qed != null && <Chip label={`QED ${mol.qed}`} size="small" variant="outlined" />}
                        </Box>
                        <Typography
                          variant="caption"
                          component="div"
                          sx={{ fontFamily: "monospace", wordBreak: "break-all", fontSize: 10, color: "text.secondary", textAlign: "center", maxHeight: 48, overflow: "hidden" }}
                        >
                          {mol.smiles}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}

        {/* ---- Undesirable Patterns Expander ---- */}
        <Accordion sx={{ mt: 3 }} defaultExpanded={false}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <WarningAmberIcon fontSize="small" color="warning" />
              <Typography fontWeight={600}>
                Undesirable Substructure Patterns (Structural Alerts)
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              These SMARTS patterns are used to filter out molecules with potentially problematic substructures.
            </Typography>
            {patternData.length === 0 ? (
              <CircularProgress size={24} />
            ) : (
              <Grid container spacing={1.5}>
                {patternData.map((p, i) => (
                  <Grid size={{ xs: 6, sm: 4, md: 3, lg: 2 }} key={i}>
                    <Box
                      sx={{
                        border: "1px solid",
                        borderColor: "warning.main",
                        borderRadius: 2,
                        p: 1,
                        textAlign: "center",
                        bgcolor: darkMode ? "rgba(255,160,0,0.06)" : "#fff8f8",
                      }}
                    >
                      {p.image ? (
                        <img
                          src={`data:image/png;base64,${p.image}`}
                          alt={p.name}
                          style={{ width: "100%", maxWidth: 140, display: "block", margin: "0 auto" }}
                        />
                      ) : (
                        <Box sx={{ height: 80, display: "flex", alignItems: "center", justifyContent: "center", color: "text.disabled" }}>
                          [No structure]
                        </Box>
                      )}
                      <Typography variant="caption" display="block" fontWeight={700} color="warning.main" sx={{ mt: 0.5 }}>
                        {p.name}
                      </Typography>
                      <Typography
                        variant="caption"
                        display="block"
                        sx={{ fontFamily: "monospace", fontSize: 9, wordBreak: "break-all", color: "text.secondary" }}
                      >
                        {p.smarts}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            )}
          </AccordionDetails>
        </Accordion>
      </Container>
    </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
