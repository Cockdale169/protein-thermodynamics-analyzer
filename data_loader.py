import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, PPBuilder
from Bio.SeqUtils import seq1
from io import StringIO


def load_pdb(uploaded_file):
    pdb_content = uploaded_file.getvalue().decode("utf-8", errors="replace")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Protein", StringIO(pdb_content))
    ppb = PPBuilder()

    results = []

    for model in structure:
        for chain in model:
            residues = [
                residue for residue in chain
                if residue.id[0] == " " and is_aa(residue, standard=False)
            ]

            if not residues:
                continue

            sequence = ""
            ca_coords = []
            res_labels = []

            for residue in residues:
                aa_code = seq1(residue.resname, custom_map={"UNK": "X"})
                sequence += aa_code

                if "CA" in residue:
                    coord = residue["CA"].get_coord()
                    ca_coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
                    res_labels.append(f"Chain {chain.id} | {residue.resname} {residue.id[1]} ({aa_code})")

            # 計算該鏈的 Ramachandran (Phi, Psi) 雙面角
            rama_data = []
            for pp in ppb.build_peptides(chain):
                phi_psi_list = pp.get_phi_psi_list()
                for res, (phi, psi) in zip(pp, phi_psi_list):
                    # 排除 N 端 (phi=None) 與 C 端 (psi=None) 缺角殘基
                    if phi is not None and psi is not None:
                        phi_deg = float(np.degrees(phi))
                        psi_deg = float(np.degrees(psi))
                        resname = res.resname.strip()
                        aa_code = seq1(resname, custom_map={"UNK": "X"})
                        rama_data.append({
                            "res_num": res.id[1],
                            "res_name": resname,
                            "aa_code": aa_code,
                            "phi": phi_deg,
                            "psi": psi_deg
                        })

            results.append({
                "model": model.id,
                "chain": chain.id,
                "residue_count": len(residues),
                "sequence": sequence,
                "ca_coords": ca_coords,
                "res_labels": res_labels,
                "rama_data": rama_data
            })

    return results