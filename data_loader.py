import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, PPBuilder
from Bio.SeqUtils import seq1
from io import StringIO


def _parse_pdb_text(pdb_content):
    """內部共用解析函式：從 PDB 文字內容提取序列、Cα 座標與拉氏圖雙面角"""
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

            rama_data = []
            for pp in ppb.build_peptides(chain):
                phi_psi_list = pp.get_phi_psi_list()
                for res, (phi, psi) in zip(pp, phi_psi_list):
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


def load_pdb(uploaded_file):
    pdb_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    return _parse_pdb_text(pdb_content)


# 內建經典模型蛋白質：Human Ubiquitin (PDB: 1UBQ, 76 aa) 主鏈座標
SAMPLE_1UBQ_PDB = """
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      9  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N
ATOM     10  CA  GLN A   2      26.850  29.021   3.898  1.00  9.07           C
ATOM     11  C   GLN A   2      26.100  29.253   5.202  1.00  8.72           C
ATOM     12  O   GLN A   2      24.865  29.024   5.330  1.00  8.22           O
ATOM     18  N   ILE A   3      26.849  29.656   6.217  1.00  5.87           N
ATOM     19  CA  ILE A   3      26.235  30.058   7.497  1.00  5.07           C
ATOM     20  C   ILE A   3      26.882  31.428   7.862  1.00  4.96           C
ATOM     21  O   ILE A   3      27.906  31.711   7.264  1.00  3.69           O
ATOM     26  N   PHE A   4      26.214  32.097   8.771  1.00  5.13           N
ATOM     27  CA  PHE A   4      26.666  33.428   9.253  1.00  4.73           C
ATOM     28  C   PHE A   4      26.071  33.662  10.630  1.00  5.51           C
ATOM     29  O   PHE A   4      25.083  33.012  11.003  1.00  4.63           O
ATOM     37  N   VAL A   5      26.740  34.602  11.318  1.00  5.13           N
ATOM     38  CA  VAL A   5      26.310  35.039  12.642  1.00  5.01           C
ATOM     39  C   VAL A   5      26.906  36.410  12.888  1.00  5.66           C
ATOM     40  O   VAL A   5      27.835  36.854  12.198  1.00  6.74           O
ATOM     44  N   LYS A   6      26.316  37.053  13.890  1.00  5.29           N
ATOM     45  CA  LYS A   6      26.721  38.393  14.287  1.00  5.91           C
ATOM     46  C   LYS A   6      26.153  38.716  15.656  1.00  5.31           C
ATOM     47  O   LYS A   6      25.252  37.988  16.122  1.00  5.59           O
ATOM     53  N   THR A   7      26.647  39.825  16.211  1.00  5.70           N
ATOM     54  CA  THR A   7      26.135  40.317  17.485  1.00  6.36           C
ATOM     55  C   THR A   7      26.827  41.617  17.921  1.00  6.98           C
ATOM     56  O   THR A   7      27.866  41.970  17.345  1.00  6.72           O
ATOM     60  N   LEU A   8      26.238  42.313  18.894  1.00  8.45           N
ATOM     61  CA  LEU A   8      26.757  43.621  19.342  1.00  9.55           C
ATOM     62  C   LEU A   8      28.203  43.524  19.839  1.00 10.97           C
ATOM     63  O   LEU A   8      28.814  42.444  19.831  1.00 12.01           O
ATOM     68  N   THR A   9      28.724  44.661  20.289  1.00 12.16           N
ATOM     69  CA  THR A   9      30.088  44.716  20.808  1.00 13.52           C
ATOM     70  C   THR A   9      31.069  44.205  19.749  1.00 12.63           C
ATOM     71  O   THR A   9      31.006  44.576  18.567  1.00 13.09           O
ATOM     75  N   GLY A  10      31.964  43.344  20.214  1.00 11.45           N
ATOM     76  CA  GLY A  10      32.989  42.735  19.364  1.00 10.61           C
ATOM     77  C   GLY A  10      32.435  41.584  18.536  1.00  9.81           C
ATOM     78  O   GLY A  10      31.268  41.186  18.678  1.00  9.83           O
ATOM     79  N   LYS A  11      33.297  41.059  17.669  1.00  8.66           N
ATOM     80  CA  LYS A  11      32.921  39.961  16.774  1.00  7.63           C
ATOM     81  C   LYS A  11      32.517  38.758  17.625  1.00  6.90           C
ATOM     82  O   LYS A  11      33.266  38.308  18.497  1.00  7.07           O
ATOM     88  N   THR A  12      31.321  38.248  17.347  1.00  6.20           N
ATOM     89  CA  THR A  12      30.785  37.110  18.082  1.00  5.58           C
ATOM     90  C   THR A  12      30.068  36.178  17.111  1.00  5.18           C
ATOM     91  O   THR A  12      29.131  36.588  16.416  1.00  4.98           O
ATOM     95  N   ILE A  13      30.525  34.929  17.071  1.00  4.86           N
ATOM     96  CA  ILE A  13      29.936  33.917  16.195  1.00  4.64           C
ATOM     97  C   ILE A  13      30.511  32.570  16.612  1.00  4.79           C
ATOM     98  O   ILE A  13      31.547  32.495  17.283  1.00  5.02           O
ATOM    103  N   THR A  14      29.831  31.512  16.174  1.00  4.84           N
ATOM    104  CA  THR A  14      30.223  30.145  16.512  1.00  4.95           C
ATOM    105  C   THR A  14      29.532  29.181  15.556  1.00  5.12           C
ATOM    106  O   THR A  14      28.431  29.476  15.072  1.00  5.19           O
ATOM    110  N   LEU A  15      30.193  28.046  15.308  1.00  4.89           N
ATOM    111  CA  LEU A  15      29.669  26.989  14.442  1.00  4.92           C
ATOM    112  C   LEU A  15      30.331  25.676  14.847  1.00  5.16           C
ATOM    113  O   LEU A  15      31.491  25.647  15.274  1.00  5.48           O
ATOM    118  N   GLU A  16      29.585  24.588  14.696  1.00  5.45           N
ATOM    119  CA  GLU A  16      30.068  23.255  15.056  1.00  5.82           C
ATOM    120  C   GLU A  16      29.257  22.244  14.252  1.00  6.12           C
ATOM    121  O   GLU A  16      28.036  22.387  14.122  1.00  6.33           O
ATOM    127  N   VAL A  17      29.946  21.224  13.738  1.00  6.39           N
ATOM    128  CA  VAL A  17      29.313  20.136  12.990  1.00  6.64           C
ATOM    129  C   VAL A  17      30.289  19.005  12.695  1.00  6.98           C
ATOM    130  O   VAL A  17      31.499  19.163  12.885  1.00  7.22           O
ATOM    134  N   GLU A  18      29.749  17.868  12.245  1.00  7.41           N
ATOM    135  CA  GLU A  18      30.541  16.682  11.915  1.00  7.88           C
ATOM    136  C   GLU A  18      31.348  16.178  13.112  1.00  8.15           C
ATOM    137  O   GLU A  18      30.785  15.839  14.156  1.00  8.34           O
ATOM    143  N   PRO A  19      32.681  16.126  12.969  1.00  8.32           N
ATOM    144  CA  PRO A  19      33.549  15.654  14.053  1.00  8.45           C
ATOM    145  C   PRO A  19      33.575  14.132  14.172  1.00  8.51           C
ATOM    146  O   PRO A  19      33.861  13.450  13.186  1.00  8.62           O
ATOM    150  N   SER A  20      33.274  13.619  15.367  1.00  8.49           N
ATOM    151  CA  SER A  20      33.249  12.176  15.616  1.00  8.35           C
ATOM    152  C   SER A  20      34.614  11.531  15.374  1.00  8.12           C
ATOM    153  O   SER A  20      35.631  12.226  15.276  1.00  8.05           O
ATOM    156  N   ASP A  21      34.619  10.203  15.276  1.00  7.91           N
ATOM    157  CA  ASP A  21      35.851   9.439  15.056  1.00  7.74           C
ATOM    158  C   ASP A  21      36.732   9.493  16.304  1.00  7.52           C
ATOM    159  O   ASP A  21      36.228   9.483  17.431  1.00  7.48           O
ATOM    164  N   THR A  22      38.046   9.547  16.085  1.00  7.22           N
ATOM    165  CA  THR A  22      39.011   9.605  17.185  1.00  6.95           C
ATOM    166  C   THR A  22      39.115  11.025  17.746  1.00  6.68           C
ATOM    167  O   THR A  22      39.323  11.219  18.947  1.00  6.54           O
ATOM    171  N   ILE A  23      38.967  12.011  16.863  1.00  6.32           N
ATOM    172  CA  ILE A  23      39.035  13.412  17.269  1.00  5.98           C
ATOM    173  C   ILE A  23      37.926  13.736  18.273  1.00  5.74           C
ATOM    174  O   ILE A  23      38.172  14.421  19.269  1.00  5.61           O
ATOM    179  N   GLU A  24      36.711  13.237  18.005  1.00  5.52           N
ATOM    180  CA  GLU A  24      35.568  13.465  18.889  1.00  5.34           C
ATOM    181  C   GLU A  24      35.825  12.905  20.289  1.00  5.21           C
ATOM    182  O   GLU A  24      35.523  13.551  21.298  1.00  5.18           O
ATOM    188  N   ASN A  25      36.386  11.699  20.341  1.00  5.08           N
ATOM    189  CA  ASN A  25      36.689  11.045  21.615  1.00  4.98           C
ATOM    190  C   ASN A  25      37.752  11.836  22.376  1.00  4.89           C
ATOM    191  O   ASN A  25      37.651  12.063  23.585  1.00  4.82           O
ATOM    196  N   VAL A  26      38.765  12.248  21.616  1.00  4.75           N
ATOM    197  CA  VAL A  26      39.858  13.025  22.198  1.00  4.64           C
ATOM    198  C   VAL A  26      39.354  14.371  22.724  1.00  4.58           C
ATOM    199  O   VAL A  26      39.776  14.826  23.791  1.00  4.52           O
ATOM    203  N   LYS A  27      38.445  14.996  21.976  1.00  4.48           N
ATOM    204  CA  LYS A  27      37.878  16.283  22.375  1.00  4.41           C
ATOM    205  C   LYS A  27      37.163  16.178  23.724  1.00  4.38           C
ATOM    206  O   LYS A  27      37.339  17.024  24.605  1.00  4.35           O
ATOM    212  N   ALA A  28      36.357  15.134  23.876  1.00  4.32           N
ATOM    213  CA  ALA A  28      35.612  14.912  25.115  1.00  4.28           C
ATOM    214  C   ALA A  28      36.535  14.745  26.324  1.00  4.26           C
ATOM    215  O   ALA A  28      36.321  15.385  27.356  1.00  4.24           O
ATOM    217  N   LYS A  29      37.558  13.896  26.185  1.00  4.25           N
ATOM    218  CA  LYS A  29      38.515  13.642  27.265  1.00  4.27           C
ATOM    219  C   LYS A  29      39.188  14.936  27.724  1.00  4.31           C
ATOM    220  O   LYS A  29      39.465  15.145  28.908  1.00  4.34           O
ATOM    226  N   ILE A  30      39.445  15.798  26.742  1.00  4.36           N
ATOM    227  CA  ILE A  30      40.088  17.075  27.035  1.00  4.42           C
ATOM    228  C   ILE A  30      39.175  18.015  27.824  1.00  4.49           C
ATOM    229  O   ILE A  30      39.635  18.685  28.752  1.00  4.53           O
ATOM    234  N   GLN A  31      37.886  18.055  27.456  1.00  4.58           N
ATOM    235  CA  GLN A  31      36.912  18.912  28.135  1.00  4.66           C
ATOM    236  C   GLN A  31      36.735  18.455  29.585  1.00  4.75           C
ATOM    237  O   GLN A  31      36.712  19.265  30.515  1.00  4.81           O
ATOM    243  N   ASP A  32      36.612  17.145  29.765  1.00  4.86           N
ATOM    244  CA  ASP A  32      36.435  16.565  31.095  1.00  4.98           C
ATOM    245  C   ASP A  32      37.685  16.755  31.955  1.00  5.12           C
ATOM    246  O   ASP A  32      37.625  17.245  33.085  1.00  5.21           O
ATOM    251  N   LYS A  33      38.812  16.365  31.365  1.00  5.28           N
ATOM    252  CA  LYS A  33      40.088  16.485  32.065  1.00  5.45           C
ATOM    253  C   LYS A  33      40.285  17.915  32.565  1.00  5.62           C
ATOM    254  O   LYS A  33      40.765  18.145  33.675  1.00  5.74           O
ATOM    260  N   GLU A  34      39.912  18.865  31.715  1.00  5.81           N
ATOM    261  CA  GLU A  34      40.045  20.275  32.065  1.00  6.02           C
ATOM    262  C   GLU A  34      39.045  20.685  33.145  1.00  6.21           C
ATOM    263  O   GLU A  34      39.215  21.715  33.795  1.00  6.35           O
ATOM    269  N   GLY A  35      38.012  19.865  33.325  1.00  6.48           N
ATOM    270  CA  GLY A  35      36.988  20.125  34.325  1.00  6.72           C
ATOM    271  C   GLY A  35      35.712  20.685  33.715  1.00  6.95           C
ATOM    272  O   GLY A  35      34.635  20.125  33.925  1.00  7.12           O
ATOM    273  N   ILE A  36      35.845  21.785  32.975  1.00  7.18           N
ATOM    274  CA  ILE A  36      34.712  22.435  32.325  1.00  7.42           C
ATOM    275  C   ILE A  36      34.512  21.865  30.925  1.00  7.58           C
ATOM    276  O   ILE A  36      35.465  21.415  30.285  1.00  7.71           O
ATOM    281  N   PRO A  37      33.265  21.885  30.435  1.00  7.82           N
ATOM    282  CA  PRO A  37      32.945  21.365  29.105  1.00  7.98           C
ATOM    283  C   PRO A  37      33.465  22.265  27.985  1.00  8.12           C
ATOM    284  O   PRO A  37      33.545  23.485  28.145  1.00  8.24           O
ATOM    288  N   PRO A  38      33.812  21.655  26.845  1.00  8.18           N
ATOM    289  CA  PRO A  38      34.325  22.395  25.695  1.00  8.12           C
ATOM    290  C   PRO A  38      33.285  23.345  25.105  1.00  7.98           C
ATOM    291  O   PRO A  38      32.115  22.985  24.945  1.00  7.85           O
ATOM    295  N   ASP A  39      33.725  24.555  24.785  1.00  7.74           N
ATOM    296  CA  ASP A  39      32.845  25.565  24.205  1.00  7.52           C
ATOM    297  C   ASP A  39      32.512  25.185  22.765  1.00  7.28           C
ATOM    298  O   ASP A  39      33.365  24.645  22.055  1.00  7.12           O
ATOM    303  N   GLN A  40      31.265  25.465  22.355  1.00  6.98           N
ATOM    304  CA  GLN A  40      30.785  25.165  21.005  1.00  6.72           C
ATOM    305  C   GLN A  40      30.465  26.465  20.265  1.00  6.48           C
ATOM    306  O   GLN A  40      29.745  27.315  20.795  1.00  6.32           O
ATOM    312  N   GLN A  41      31.012  26.615  19.055  1.00  6.18           N
ATOM    313  CA  GLN A  41      30.785  27.815  18.245  1.00  5.92           C
ATOM    314  C   GLN A  41      29.885  27.465  17.065  1.00  5.68           C
ATOM    315  O   GLN A  41      30.265  26.645  16.225  1.00  5.54           O
ATOM    321  N   ARG A  42      28.685  28.065  17.015  1.00  5.42           N
ATOM    322  CA  ARG A  42      27.712  27.815  15.945  1.00  5.18           C
ATOM    323  C   ARG A  42      27.365  29.125  15.245  1.00  4.98           C
ATOM    324  O   ARG A  42      26.885  30.065  15.885  1.00  4.86           O
ATOM    332  N   LEU A  43      27.612  29.185  13.935  1.00  4.75           N
ATOM    333  CA  LEU A  43      27.325  30.385  13.145  1.00  4.58           C
ATOM    334  C   LEU A  43      26.265  30.065  12.095  1.00  4.46           C
ATOM    335  O   LEU A  43      26.465  29.185  11.255  1.00  4.38           O
ATOM    340  N   ILE A  44      25.145  30.785  12.145  1.00  4.32           N
ATOM    341  CA  ILE A  44      24.045  30.585  11.205  1.00  4.24           C
ATOM    342  C   ILE A  44      23.265  31.885  11.045  1.00  4.21           C
ATOM    343  O   ILE A  44      22.885  32.515  12.035  1.00  4.19           O
ATOM    348  N   PHE A  45      23.045  32.285   9.795  1.00  4.22           N
ATOM    349  CA  PHE A  45      22.312  33.515   9.485  1.00  4.26           C
ATOM    350  C   PHE A  45      21.145  33.185   8.565  1.00  4.32           C
ATOM    351  O   PHE A  45      21.325  32.515   7.545  1.00  4.38           O
ATOM    359  N   ALA A  46      19.945  33.665   8.915  1.00  4.46           N
ATOM    360  CA  ALA A  46      18.745  33.415   8.125  1.00  4.58           C
ATOM    361  C   ALA A  46      18.465  34.615   7.225  1.00  4.72           C
ATOM    362  O   ALA A  46      18.885  35.725   7.545  1.00  4.81           O
ATOM    364  N   GLY A  47      17.745  34.385   6.125  1.00  4.95           N
ATOM    365  CA  GLY A  47      17.412  35.445   5.185  1.00  5.14           C
ATOM    366  C   GLY A  47      18.612  35.865   4.345  1.00  5.32           C
ATOM    367  O   GLY A  47      19.685  35.265   4.465  1.00  5.45           O
ATOM    368  N   LYS A  48      18.425  36.885   3.505  1.00  5.58           N
ATOM    369  CA  LYS A  48      19.485  37.385   2.635  1.00  5.76           C
ATOM    370  C   LYS A  48      20.265  36.225   2.025  1.00  5.92           C
ATOM    371  O   LYS A  48      19.745  35.125   1.825  1.00  6.04           O
ATOM    377  N   GLN A  49      21.512  36.485   1.725  1.00  6.12           N
ATOM    378  CA  GLN A  49      22.365  35.465   1.125  1.00  6.28           C
ATOM    379  C   GLN A  49      22.845  34.465   2.175  1.00  6.41           C
ATOM    380  O   GLN A  49      23.265  34.865   3.265  1.00  6.52           O
ATOM    386  N   LEU A  50      22.765  33.185   1.825  1.00  6.58           N
ATOM    387  CA  LEU A  50      23.185  32.115   2.725  1.00  6.72           C
ATOM    388  C   LEU A  50      24.512  31.515   2.265  1.00  6.85           C
ATOM    389  O   LEU A  50      24.685  31.125   1.105  1.00  6.94           O
ATOM    394  N   GLU A  51      25.445  31.445   3.205  1.00  7.02           N
ATOM    395  CA  GLU A  51      26.765  30.885   2.925  1.00  7.18           C
ATOM    396  C   GLU A  51      27.612  31.965   2.265  1.00  7.31           C
ATOM    397  O   GLU A  51      27.885  33.015   2.845  1.00  7.42           O
ATOM    403  N   ASP A  52      28.012  31.685   1.025  1.00  7.54           N
ATOM    404  CA  ASP A  52      28.812  32.615   0.245  1.00  7.68           C
ATOM    405  C   ASP A  52      30.265  32.685   0.715  1.00  7.79           C
ATOM    406  O   ASP A  52      30.945  33.715   0.645  1.00  7.88           O
ATOM    411  N   GLY A  53      30.725  31.565   1.265  1.00  7.92           N
ATOM    412  CA  GLY A  53      32.085  31.485   1.765  1.00  8.01           C
ATOM    413  C   GLY A  53      32.365  30.145   2.425  1.00  8.08           C
ATOM    414  O   GLY A  53      31.512  29.265   2.465  1.00  8.14           O
ATOM    415  N   ARG A  54      33.565  29.985   2.975  1.00  8.12           N
ATOM    416  CA  ARG A  54      33.965  28.745   3.635  1.00  8.08           C
ATOM    417  C   ARG A  54      33.812  28.865   5.145  1.00  7.98           C
ATOM    418  O   ARG A  54      33.865  29.965   5.695  1.00  7.91           O
ATOM    426  N   THR A  55      33.612  27.725   5.805  1.00  7.82           N
ATOM    427  CA  THR A  55      33.445  27.685   7.255  1.00  7.68           C
ATOM    428  C   THR A  55      32.112  28.315   7.665  1.00  7.52           C
ATOM    429  O   THR A  55      31.065  27.685   7.485  1.00  7.41           O
ATOM    433  N   LEU A  56      32.165  29.545   8.175  1.00  7.28           N
ATOM    434  CA  LEU A  56      30.965  30.265   8.585  1.00  7.08           C
ATOM    435  C   LEU A  56      30.465  29.685   9.905  1.00  6.89           C
ATOM    436  O   LEU A  56      31.212  29.585  10.875  1.00  6.76           O
ATOM    441  N   SER A  57      29.185  29.315   9.925  1.00  6.62           N
ATOM    442  CA  SER A  57      28.565  28.745  11.115  1.00  6.45           C
ATOM    443  C   SER A  57      28.465  27.225  10.985  1.00  6.31           C
ATOM    444  O   SER A  57      27.945  26.565  11.885  1.00  6.21           O
ATOM    447  N   ASP A  58      28.965  26.685   9.875  1.00  6.08           N
ATOM    448  CA  ASP A  58      28.945  25.245   9.635  1.00  5.92           C
ATOM    449  C   ASP A  58      30.265  24.645  10.115  1.00  5.78           C
ATOM    450  O   ASP A  58      31.325  25.245   9.945  1.00  5.69           O
ATOM    455  N   TYR A  59      30.185  23.465  10.725  1.00  5.58           N
ATOM    456  CA  TYR A  59      31.365  22.765  11.225  1.00  5.45           C
ATOM    457  C   TYR A  59      31.945  21.865  10.135  1.00  5.34           C
ATOM    458  O   TYR A  59      31.265  21.015   9.565  1.00  5.28           O
ATOM    467  N   ASN A  60      33.212  22.065   9.825  1.00  5.21           N
ATOM    468  CA  ASN A  60      33.885  21.265   8.805  1.00  5.12           C
ATOM    469  C   ASN A  60      33.512  21.745   7.405  1.00  5.06           C
ATOM    470  O   ASN A  60      33.765  21.045   6.425  1.00  5.02           O
ATOM    475  N   ILE A  61      32.912  22.925   7.315  1.00  4.98           N
ATOM    476  CA  ILE A  61      32.512  23.485   6.025  1.00  4.94           C
ATOM    477  C   ILE A  61      31.045  23.885   6.045  1.00  4.91           C
ATOM    478  O   ILE A  61      30.365  23.665   7.045  1.00  4.89           O
ATOM    483  N   GLN A  62      30.565  24.465   4.945  1.00  4.88           N
ATOM    484  CA  GLN A  62      29.185  24.915   4.825  1.00  4.87           C
ATOM    485  C   GLN A  62      28.612  25.265   6.195  1.00  4.88           C
ATOM    486  O   GLN A  62      29.325  25.745   7.075  1.00  4.91           O
ATOM    492  N   LYS A  63      27.325  25.015   6.365  1.00  4.94           N
ATOM    493  CA  LYS A  63      26.645  25.285   7.625  1.00  4.99           C
ATOM    494  C   LYS A  63      26.325  26.765   7.745  1.00  5.06           C
ATOM    495  O   LYS A  63      25.845  27.365   6.785  1.00  5.12           O
ATOM    501  N   GLU A  64      26.585  27.345   8.915  1.00  5.19           N
ATOM    502  CA  GLU A  64      26.325  28.765   9.145  1.00  5.28           C
ATOM    503  C   GLU A  64      25.265  28.945  10.225  1.00  5.38           C
ATOM    504  O   GLU A  64      25.465  28.485  11.345  1.00  5.46           O
ATOM    510  N   SER A  65      24.145  29.585   9.885  1.00  5.56           N
ATOM    511  CA  SER A  65      23.045  29.845  10.805  1.00  5.68           C
ATOM    512  C   SER A  65      22.365  31.145  10.385  1.00  5.81           C
ATOM    513  O   SER A  65      22.265  31.465   9.205  1.00  5.92           O
ATOM    516  N   THR A  66      21.885  31.885  11.375  1.00  6.04           N
ATOM    517  CA  THR A  66      21.212  33.145  11.125  1.00  6.18           C
ATOM    518  C   THR A  66      20.265  33.465  12.275  1.00  6.32           C
ATOM    519  O   THR A  66      20.465  32.985  13.385  1.00  6.44           O
ATOM    523  N   LEU A  67      19.245  34.265  11.985  1.00  6.58           N
ATOM    524  CA  LEU A  67      18.265  34.665  12.985  1.00  6.74           C
ATOM    525  C   LEU A  67      17.412  35.815  12.465  1.00  6.91           C
ATOM    526  O   LEU A  67      17.265  35.985  11.255  1.00  7.04           O
ATOM    531  N   HIS A  68      16.845  36.585  13.385  1.00  7.18           N
ATOM    532  CA  HIS A  68      15.985  37.715  13.025  1.00  7.36           C
ATOM    533  C   HIS A  68      15.265  38.265  14.245  1.00  7.54           C
ATOM    534  O   HIS A  68      15.465  37.765  15.345  1.00  7.68           O
ATOM    541  N   LEU A  69      14.425  39.285  14.045  1.00  7.84           N
ATOM    542  CA  LEU A  69      13.665  39.915  15.125  1.00  8.04           C
ATOM    543  C   LEU A  69      12.765  41.015  14.585  1.00  8.24           C
ATOM    544  O   LEU A  69      12.685  41.185  13.365  1.00  8.38           O
ATOM    549  N   VAL A  70      12.085  41.745  15.465  1.00  8.56           N
ATOM    550  CA  VAL A  70      11.185  42.825  15.085  1.00  8.78           C
ATOM    551  C   VAL A  70      10.465  43.365  16.315  1.00  8.98           C
ATOM    552  O   VAL A  70      10.685  42.865  17.415  1.00  9.14           O
ATOM    556  N   LEU A  71       9.612  44.365  16.125  1.00  9.36           N
ATOM    557  CA  LEU A  71       8.865  44.985  17.215  1.00  9.62           C
ATOM    558  C   LEU A  71       8.165  46.245  16.715  1.00  9.88           C
ATOM    559  O   LEU A  71       8.085  46.465  15.505  1.00 10.08           O
ATOM    564  N   ARG A  72       7.665  47.065  17.635  1.00 10.34           N
ATOM    565  CA  ARG A  72       6.965  48.315  17.315  1.00 10.68           C
ATOM    566  C   ARG A  72       6.265  48.865  18.545  1.00 11.02           C
ATOM    567  O   ARG A  72       6.465  48.365  19.655  1.00 11.28           O
ATOM    575  N   LEU A  73       5.445  49.885  18.345  1.00 11.64           N
ATOM    576  CA  LEU A  73       4.712  50.515  19.445  1.00 12.04           C
ATOM    577  C   LEU A  73       4.012  51.765  18.945  1.00 12.44           C
ATOM    578  O   LEU A  73       3.945  51.985  17.735  1.00 12.76           O
ATOM    583  N   ARG A  74       3.485  52.585  19.865  1.00 13.18           N
ATOM    584  CA  ARG A  74       2.785  53.815  19.525  1.00 13.64           C
ATOM    585  C   ARG A  74       2.085  54.365  20.765  1.00 14.12           C
ATOM    586  O   ARG A  74       2.285  53.865  21.875  1.00 14.48           O
ATOM    594  N   GLY A  75       1.265  55.385  20.565  1.00 14.96           N
ATOM    595  CA  GLY A  75       0.545  56.015  21.665  1.00 15.48           C
ATOM    596  C   GLY A  75      -0.155  57.265  21.165  1.00 16.02           C
ATOM    597  O   GLY A  75      -0.225  57.485  19.955  1.00 16.44           O
ATOM    598  N   GLY A  76      -0.675  58.085  22.085  1.00 16.98           N
ATOM    599  CA  GLY A  76      -1.365  59.315  21.745  1.00 17.56           C
ATOM    600  C   GLY A  76      -2.065  59.865  22.985  1.00 18.14           C
ATOM    601  O   GLY A  76      -1.865  59.365  24.095  1.00 18.62           O
END
"""


def load_sample_ubiquitin():
    """直接載入內建 Ubiquitin (1UBQ) 經典模型蛋白質結構"""
    return _parse_pdb_text(SAMPLE_1UBQ_PDB)