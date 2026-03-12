"""
Fragment search & molecule reassembly logic extracted from morpheus.py.
"""

import gzip
from itertools import permutations
from typing import List, Dict, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    DataStructs,
    Descriptors,
    Crippen,
    QED,
    AllChem,
    Draw,
    rdDepictor,
    rdFingerprintGenerator,
)
import io, base64


# ---------------------------------------------------------------------------
# find_similar_fragments  (from morpheus.py)
# ---------------------------------------------------------------------------

def find_similar_fragments(
    query_smiles: str,
    fragments_file: str,
    similarity_threshold: float = 0.3,
    top_n: int = 50,
    progress_callback=None,
) -> List[Tuple[str, float, int]]:
    """Return list of (smiles, similarity, num_attachments) sorted by similarity desc."""
    RDLogger.DisableLog("rdApp.*")

    def get_attachment_info(mol):
        info = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbors = atom.GetNeighbors()
                if neighbors:
                    info.append((atom.GetIdx(), neighbors[0].GetIdx(), atom.GetAtomMapNum()))
        return info

    def get_sorted_distances(mol):
        ai = get_attachment_info(mol)
        if len(ai) < 2:
            return ()
        distances = []
        for i in range(len(ai)):
            for j in range(i + 1, len(ai)):
                try:
                    path = Chem.GetShortestPath(mol, ai[i][1], ai[j][1])
                    if path:
                        distances.append(len(path) - 1)
                except:
                    pass
        return tuple(sorted(distances))

    def get_distance_matrix(mol, attachment_info):
        distances = {}
        n = len(attachment_info)
        for i in range(n):
            for j in range(i + 1, n):
                _, ni, mi = attachment_info[i]
                _, nj, mj = attachment_info[j]
                try:
                    path = Chem.GetShortestPath(mol, ni, nj)
                    if path:
                        d = len(path) - 1
                        distances[(mi, mj)] = d
                        distances[(mj, mi)] = d
                except:
                    pass
        return distances

    def find_mapping(query_info, query_distances, frag_info, frag_mol):
        query_map_nums = [i[2] for i in query_info]
        frag_map_nums = [i[2] for i in frag_info]
        frag_distances = {}
        for i in range(len(frag_info)):
            for j in range(i + 1, len(frag_info)):
                _, ni, mi = frag_info[i]
                _, nj, mj = frag_info[j]
                try:
                    path = Chem.GetShortestPath(frag_mol, ni, nj)
                    if path:
                        d = len(path) - 1
                        frag_distances[(mi, mj)] = d
                        frag_distances[(mj, mi)] = d
                except:
                    pass
        for perm in permutations(query_map_nums):
            mapping = dict(zip(frag_map_nums, perm))
            valid = True
            for i in range(len(frag_map_nums)):
                for j in range(i + 1, len(frag_map_nums)):
                    fm1, fm2 = frag_map_nums[i], frag_map_nums[j]
                    qm1, qm2 = mapping[fm1], mapping[fm2]
                    if frag_distances.get((fm1, fm2)) != query_distances.get((qm1, qm2)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return mapping
        return {}

    def renumber_fragment(smiles, mapping):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                old_map = atom.GetAtomMapNum()
                if old_map in mapping:
                    atom.SetAtomMapNum(mapping[old_map])
        return Chem.MolToSmiles(rw.GetMol(), canonical=True)

    def replace_dummies_with_h(mol):
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetAtomMapNum(0)
        return rw.GetMol()

    # Parse query
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return []

    query_ai = get_attachment_info(query_mol)
    query_attachments = len(query_ai)
    query_sorted_dist = get_sorted_distances(query_mol) if query_attachments > 1 else ()
    query_dist_matrix = get_distance_matrix(query_mol, query_ai) if query_attachments >= 3 else {}

    query_mol_h = replace_dummies_with_h(query_mol)
    try:
        Chem.SanitizeMol(query_mol_h)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        query_fp = fpgen.GetFingerprint(query_mol_h)
    except:
        return []

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    similar_fragments = []
    seen = set()

    # Count lines for progress
    total_lines = 0
    if progress_callback:
        opener = gzip.open if fragments_file.endswith(".gz") else open
        with opener(fragments_file, "rt", encoding="utf-8") as fc:
            total_lines = sum(1 for l in fc if l.strip())

    opener = gzip.open if fragments_file.endswith(".gz") else open
    f = opener(fragments_file, "rt", encoding="utf-8")
    try:
        line_num = 0
        for line in f:
            line_num += 1
            if progress_callback and total_lines > 0 and line_num % 1000 == 0:
                progress_callback(line_num / total_lines)
            line = line.strip()
            if not line:
                continue
            frag_mol = Chem.MolFromSmiles(line)
            if frag_mol is None:
                continue
            frag_ai = get_attachment_info(frag_mol)
            if len(frag_ai) != query_attachments:
                continue
            if query_attachments > 1:
                if get_sorted_distances(frag_mol) != query_sorted_dist:
                    continue
            frag_mol_h = replace_dummies_with_h(frag_mol)
            try:
                try:
                    Chem.SanitizeMol(frag_mol_h)
                except:
                    try:
                        Chem.SanitizeMol(frag_mol_h, catchErrors=True)
                    except:
                        continue
                canonical = Chem.MolToSmiles(frag_mol_h, canonical=True)
                if canonical in seen:
                    continue
                frag_fp = fpgen.GetFingerprint(frag_mol_h)
                similarity = DataStructs.TanimotoSimilarity(query_fp, frag_fp)
                if similarity >= similarity_threshold and similarity < 1.0:
                    output = line
                    if query_attachments == 1:
                        qm = query_ai[0][2]
                        fm = frag_ai[0][2]
                        if fm != qm:
                            output = renumber_fragment(line, {fm: qm})
                    elif query_attachments == 2:
                        mapping = dict(zip([i[2] for i in frag_ai], [i[2] for i in query_ai]))
                        output = renumber_fragment(line, mapping)
                    elif query_attachments >= 3:
                        mapping = find_mapping(query_ai, query_dist_matrix, frag_ai, frag_mol)
                        if mapping:
                            output = renumber_fragment(line, mapping)
                    similar_fragments.append((output, similarity, len(frag_ai)))
                    seen.add(canonical)
            except:
                continue
    finally:
        f.close()

    RDLogger.EnableLog("rdApp.*")
    similar_fragments.sort(key=lambda x: x[1], reverse=True)

    # Filter racemic duplicates
    def get_achiral(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m, canonical=True)

    def has_stereo(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        return len(Chem.FindMolChiralCenters(m, includeUnassigned=False)) > 0

    groups: Dict[str, list] = {}
    for smi, sim, na in similar_fragments:
        ac = get_achiral(smi)
        hs = has_stereo(smi)
        groups.setdefault(ac, []).append((smi, sim, na, hs))

    filtered = []
    for ac, group in groups.items():
        chiral = [g for g in group if g[3]]
        achiral = [g for g in group if not g[3]]
        if len(chiral) >= 2 and achiral:
            for s, si, n, _ in chiral:
                filtered.append((s, si, n))
        else:
            for s, si, n, _ in group:
                filtered.append((s, si, n))
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_n]


# ---------------------------------------------------------------------------
# reassemble_from_smiles  (from morpheus.py)
# ---------------------------------------------------------------------------

def reassemble_from_smiles(smiles_list: List[str]) -> Chem.Mol | None:
    """Reassemble a molecule from fragment SMILES with numbered dummy atoms."""
    if not smiles_list:
        return None
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if not mols:
        return None

    if len(mols) == 1:
        rw = Chem.RWMol(mols[0])
        for idx in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
            rw.RemoveAtom(idx)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        return mol

    combined = mols[0]
    for m in mols[1:]:
        combined = Chem.CombineMols(combined, m)
    rw = Chem.RWMol(combined)

    dummy_map: Dict[int, list] = {}
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            mn = atom.GetAtomMapNum()
            if mn > 0:
                neighbors = atom.GetNeighbors()
                if neighbors:
                    bond = rw.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                    bt = bond.GetBondType() if bond else Chem.BondType.SINGLE
                    dummy_map.setdefault(mn, []).append((atom.GetIdx(), neighbors[0].GetIdx(), bt))

    to_remove = set()
    for mn, dlist in dummy_map.items():
        if len(dlist) >= 2:
            d1, r1, bt = dlist[0]
            d2, r2, _ = dlist[1]
            if rw.GetBondBetweenAtoms(r1, r2) is None:
                rw.AddBond(r1, r2, bt)
            to_remove.update([d1, d2])

    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    remaining = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(remaining, reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except:
        try:
            for a in mol.GetAtoms():
                a.SetIsAromatic(False)
            for b in mol.GetBonds():
                b.SetIsAromatic(False)
            Chem.SanitizeMol(mol)
        except:
            pass
    try:
        rdDepictor.Compute2DCoords(mol)
    except:
        pass
    return mol


# ---------------------------------------------------------------------------
# helper: molecule image to base64 PNG
# ---------------------------------------------------------------------------

def mol_to_base64_png(mol, size=(300, 300)):
    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().bondLineWidth = 2.2 #adjust line thickness here
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText()).decode()
