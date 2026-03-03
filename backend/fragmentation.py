"""
Molecule fragmentation logic extracted from morpheus.py.
Decomposes molecules into ring and non-ring fragments with wildcard attachment points.
"""

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from typing import List, Dict, Set, Tuple
import io
import base64


def decompose_molecule_with_wildcards(
    mol: Chem.Mol,
    include_terminal_substituents: bool = True,
    preserve_fused_rings: bool = True,
    max_terminal_atoms: int = 3,
) -> Dict[str, List[Dict]]:
    """
    Decompose a molecule into its individual rings (or fused ring systems) AND non-ring fragments,
    adding numbered wildcard dummy atoms ([*:1], [*:2], etc.) at each attachment point.
    """
    if mol is None:
        return {"rings": [], "non_rings": []}

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    # Get all atoms that are part of any ring
    all_ring_atoms: Set[int] = set()
    for ring in atom_rings:
        all_ring_atoms.update(ring)

    # ============== PART 1: Process Ring Systems ==============
    if preserve_fused_rings and atom_rings:
        ring_sets = [set(ring) for ring in atom_rings]

        merged = True
        while merged:
            merged = False
            new_ring_sets: list[set[int]] = []
            used = [False] * len(ring_sets)

            for i in range(len(ring_sets)):
                if used[i]:
                    continue
                current = ring_sets[i].copy()
                used[i] = True

                for j in range(i + 1, len(ring_sets)):
                    if used[j]:
                        continue
                    if current & ring_sets[j]:
                        current |= ring_sets[j]
                        used[j] = True
                        merged = True

                new_ring_sets.append(current)

            ring_sets = new_ring_sets

        ring_systems = [tuple(sorted(rs)) for rs in ring_sets]
    else:
        ring_systems = [tuple(ring) for ring in atom_rings] if atom_rings else []

    # First pass: collect all fragments and their atoms
    all_fragments: list[tuple] = []
    atoms_assigned_to_rings: Set[int] = set()

    for ring_system in ring_systems:
        ring_atoms = set(ring_system)
        is_fused = preserve_fused_rings and len(ring_system) > 6

        if include_terminal_substituents:
            expanded_atoms = set(ring_atoms)

            def get_terminal_chain(
                start_idx: int, from_atoms: Set[int], max_atoms: int
            ) -> Set[int]:
                if start_idx in all_ring_atoms:
                    return set()

                chain: Set[int] = set()
                queue = [start_idx]
                visited = set(from_atoms)

                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    if current in all_ring_atoms:
                        return set()

                    visited.add(current)
                    chain.add(current)

                    if len(chain) > max_atoms:
                        return set()

                    atom = mol.GetAtomWithIdx(current)
                    for neighbor in atom.GetNeighbors():
                        nb_idx = neighbor.GetIdx()
                        if nb_idx not in visited:
                            queue.append(nb_idx)

                for atom_idx in chain:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        nb_idx = neighbor.GetIdx()
                        if nb_idx not in chain and nb_idx not in from_atoms:
                            if nb_idx in all_ring_atoms:
                                return set()

                return chain

            for ring_atom_idx in list(ring_atoms):
                atom = mol.GetAtomWithIdx(ring_atom_idx)
                for neighbor in atom.GetNeighbors():
                    nb_idx = neighbor.GetIdx()
                    if nb_idx in expanded_atoms:
                        continue
                    if nb_idx in all_ring_atoms:
                        continue
                    terminal_chain = get_terminal_chain(
                        nb_idx, expanded_atoms, max_terminal_atoms
                    )
                    if terminal_chain:
                        expanded_atoms.update(terminal_chain)

            ring_atoms_list = list(expanded_atoms)
        else:
            ring_atoms_list = list(ring_atoms)

        atoms_assigned_to_rings.update(ring_atoms_list)
        all_fragments.append(("ring", ring_system, set(ring_atoms_list), is_fused))

    # Get non-ring atoms
    all_atoms = set(range(mol.GetNumAtoms()))
    non_ring_atoms = all_atoms - atoms_assigned_to_rings

    if non_ring_atoms:
        visited: Set[int] = set()

        for start_atom in non_ring_atoms:
            if start_atom in visited:
                continue

            component: Set[int] = set()
            queue = [start_atom]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in non_ring_atoms:
                    continue

                visited.add(current)
                component.add(current)

                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    nb_idx = neighbor.GetIdx()
                    if nb_idx in non_ring_atoms and nb_idx not in visited:
                        queue.append(nb_idx)

            if component:
                all_fragments.append(("non_ring", None, component, False))

    # ============== Build Bond-to-Number Mapping ==============
    bond_number_map: Dict[Tuple[int, int], int] = {}
    current_bond_number = 1

    for i, (cat1, rs1, atoms1, _) in enumerate(all_fragments):
        for j, (cat2, rs2, atoms2, _) in enumerate(all_fragments):
            if i >= j:
                continue
            for a1 in atoms1:
                atom = mol.GetAtomWithIdx(a1)
                for neighbor in atom.GetNeighbors():
                    a2 = neighbor.GetIdx()
                    if a2 in atoms2:
                        bond_key = (min(a1, a2), max(a1, a2))
                        if bond_key not in bond_number_map:
                            bond_number_map[bond_key] = current_bond_number
                            current_bond_number += 1

    # ============== Process Each Fragment ==============
    def process_fragment(
        atom_set: Set[int],
        frag_category: str,
        ring_system: Tuple = None,
        is_fused: bool = False,
    ) -> Dict | None:
        atom_list = list(atom_set)

        attachment_info: list[tuple[int, int, int]] = []
        for a_idx in atom_list:
            atom = mol.GetAtomWithIdx(a_idx)
            for neighbor in atom.GetNeighbors():
                nb_idx = neighbor.GetIdx()
                if nb_idx not in atom_set:
                    bond_key = (min(a_idx, nb_idx), max(a_idx, nb_idx))
                    bond_num = bond_number_map.get(bond_key, 0)
                    attachment_info.append((a_idx, nb_idx, bond_num))

        attachment_atoms = sorted(set(a for a, _, _ in attachment_info))

        if frag_category == "non_ring":
            if len(attachment_atoms) == 0:
                frag_type = "isolated"
            elif len(attachment_atoms) == 1:
                frag_type = "terminal"
            else:
                frag_type = "linker"
        else:
            frag_type = "fused_ring" if is_fused else "ring"

        base_smi = Chem.MolFragmentToSmiles(mol, atom_list, canonical=True)

        bonds_to_break: list[int] = []
        dummy_labels: list[tuple[int, tuple[int, int]]] = []

        for a_idx, nb_idx, bond_num in attachment_info:
            bond = mol.GetBondBetweenAtoms(a_idx, nb_idx)
            if bond is not None:
                bond_idx = bond.GetIdx()
                if bond_idx not in [b for b, _ in dummy_labels]:
                    dummy_labels.append((bond_idx, (bond_num, bond_num)))
                    bonds_to_break.append(bond_idx)

        frag_mol = None
        wildcard_smi = None

        if bonds_to_break:
            try:
                dummy_label_list = []
                for bond_idx in bonds_to_break:
                    for bi, (l1, l2) in dummy_labels:
                        if bi == bond_idx:
                            dummy_label_list.append((l1, l2))
                            break

                frag_mol_temp = Chem.FragmentOnBonds(
                    mol,
                    bonds_to_break,
                    addDummies=True,
                    dummyLabels=dummy_label_list,
                )
                frags = Chem.GetMolFrags(
                    frag_mol_temp, asMols=True, sanitizeFrags=False
                )
                frag_atom_lists = Chem.GetMolFrags(frag_mol_temp, asMols=False)

                target_frag = None
                for frag, frag_atoms in zip(frags, frag_atom_lists):
                    frag_atoms_set = set(frag_atoms)
                    check_atoms = ring_system if ring_system else atom_list
                    if any(a_idx in frag_atoms_set for a_idx in check_atoms):
                        non_dummy_count = sum(
                            1 for a in frag.GetAtoms() if a.GetAtomicNum() != 0
                        )
                        if non_dummy_count == len(atom_list):
                            target_frag = frag
                            break

                if target_frag is not None:
                    frag_mol = target_frag
                    rw = Chem.RWMol(frag_mol)
                    for atom in rw.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            isotope = atom.GetIsotope()
                            if isotope > 0:
                                atom.SetAtomMapNum(isotope)
                            atom.SetIsotope(0)
                    frag_mol = rw.GetMol()
                    try:
                        Chem.SanitizeMol(frag_mol)
                    except Exception:
                        try:
                            for atom in frag_mol.GetAtoms():
                                atom.SetIsAromatic(False)
                            for bond in frag_mol.GetBonds():
                                bond.SetIsAromatic(False)
                            Chem.SanitizeMol(frag_mol)
                        except Exception:
                            frag_mol = None
            except Exception:
                frag_mol = None

        if frag_mol is None:
            try:
                frag_mol = Chem.MolFromSmiles(base_smi)
                wildcard_smi = base_smi
            except Exception:
                return None

        if frag_mol is None:
            return None

        try:
            rdDepictor.Compute2DCoords(frag_mol)
        except Exception:
            pass

        if wildcard_smi is None:
            try:
                wildcard_smi = Chem.MolToSmiles(frag_mol, canonical=True)
            except Exception:
                wildcard_smi = base_smi

        test_mol = Chem.MolFromSmiles(wildcard_smi)
        if test_mol is None:
            wildcard_smi = base_smi

        hetero_count = sum(
            1 for a in frag_mol.GetAtoms() if a.GetAtomicNum() not in (0, 1, 6)
        )

        result = {
            "base_smiles": base_smi,
            "wildcard_smiles": wildcard_smi,
            "frag_mol": frag_mol,
            "atom_indices": tuple(atom_list),
            "attachment_atoms": attachment_atoms,
            "size": len(ring_system) if ring_system else len(atom_list),
            "hetero_count": hetero_count,
            "frag_type": frag_type,
        }

        if ring_system:
            result["core_ring_atoms"] = ring_system
            result["total_atoms"] = len(atom_list)

        return result

    ring_results: list[Dict] = []
    non_ring_results: list[Dict] = []
    seen_wildcard_smiles: set[str] = set()

    for cat, ring_sys, atoms, is_fused in all_fragments:
        result = process_fragment(atoms, cat, ring_sys, is_fused)
        if result is None:
            continue
        if result["wildcard_smiles"] in seen_wildcard_smiles:
            continue
        seen_wildcard_smiles.add(result["wildcard_smiles"])
        if cat == "ring":
            ring_results.append(result)
        else:
            non_ring_results.append(result)

    return {"rings": ring_results, "non_rings": non_ring_results}


def mol_to_base64_png(mol: Chem.Mol, size: tuple[int, int] = (300, 300)) -> str:
    """Render an RDKit mol to a base64-encoded PNG string."""
    rdDepictor.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def count_heavy_atoms(smiles: str) -> int:
    """Count non-hydrogen, non-dummy atoms in a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
