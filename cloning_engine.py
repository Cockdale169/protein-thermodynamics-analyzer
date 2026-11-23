import math
import numpy as np
from datetime import datetime


# =====================================================================
# 1. 密碼子偏好表與宿主權重 (E. coli / Homo sapiens / Degenerate)
# =====================================================================
ECOLI_RNA_CODONS = {
    "A": "GCG", "R": "CGU", "N": "AAC", "D": "GAU", "C": "UGC",
    "Q": "CAG", "E": "GAA", "G": "GGU", "H": "CAC", "I": "AUU",
    "L": "CUG", "K": "AAA", "M": "AUG", "F": "UUU", "P": "CCG",
    "S": "UCU", "T": "ACC", "W": "UGG", "Y": "UAU", "V": "GUU",
    "*": "UAA", "X": "NNN"
}

HUMAN_RNA_CODONS = {
    "A": "GCC", "R": "AGA", "N": "AAC", "D": "GAC", "C": "UGC",
    "Q": "CAG", "E": "GAG", "G": "GGC", "H": "CAC", "I": "AUC",
    "L": "CUG", "K": "AAG", "M": "AUG", "F": "UUC", "P": "CCC",
    "S": "AGC", "T": "ACC", "W": "UGG", "Y": "UAC", "V": "GUG",
    "*": "UGA", "X": "NNN"
}

DEGENERATE_RNA_CODONS = {
    "A": "GCN", "R": "CGN", "N": "AAY", "D": "GAY", "C": "UGY",
    "Q": "CAR", "E": "GAR", "G": "GGN", "H": "CAY", "I": "AUH",
    "L": "CUN", "K": "AAR", "M": "AUG", "F": "UUY", "P": "CCN",
    "S": "UCN", "T": "ACN", "W": "UGG", "Y": "UAY", "V": "GUN",
    "*": "UAR", "X": "NNN"
}

CODON_WEIGHTS = {
    "ecoli": {
        "GCG": 1.0, "GCC": 0.65, "GCU": 0.45, "GCA": 0.50,
        "CGU": 1.0, "CGC": 0.90, "CGG": 0.15, "CGA": 0.10, "AGA": 0.08, "AGG": 0.05,
        "AAC": 1.0, "AAU": 0.45, "GAU": 1.0, "GAC": 0.65,
        "UGC": 1.0, "UGU": 0.75, "CAG": 1.0, "CAA": 0.40,
        "GAA": 1.0, "GAG": 0.45, "GGU": 1.0, "GGC": 0.85, "GGG": 0.25, "GGA": 0.15,
        "CAC": 1.0, "CAU": 0.70, "AUU": 1.0, "AUC": 0.85, "AUA": 0.10,
        "CUG": 1.0, "CUC": 0.20, "CUU": 0.20, "CUA": 0.08, "UUG": 0.25, "UUA": 0.20,
        "AAA": 1.0, "AAG": 0.35, "AUG": 1.0, "UUU": 1.0, "UUC": 0.80,
        "CCG": 1.0, "CCA": 0.35, "CCU": 0.25, "CCC": 0.12,
        "UCU": 1.0, "UCC": 0.85, "AGC": 0.80, "AGU": 0.30, "UCA": 0.25, "UCG": 0.30,
        "ACC": 1.0, "ACU": 0.40, "ACG": 0.55, "ACA": 0.20,
        "UGG": 1.0, "UAU": 1.0, "UAC": 0.80,
        "GUU": 1.0, "GUG": 0.90, "GUC": 0.40, "GUA": 0.45
    },
    "human": {
        "GCC": 1.0, "GCU": 0.65, "GCA": 0.55, "GCG": 0.25,
        "AGA": 1.0, "AGG": 0.95, "CGG": 0.90, "CGC": 0.85, "CGA": 0.50, "CGU": 0.35,
        "AAC": 1.0, "AAU": 0.85, "GAC": 1.0, "GAU": 0.85,
        "UGC": 1.0, "UGU": 0.80, "CAG": 1.0, "CAA": 0.35,
        "GAG": 1.0, "GAA": 0.70, "GGC": 1.0, "GGG": 0.75, "GGA": 0.70, "GGU": 0.45,
        "CAC": 1.0, "CAU": 0.70, "AUC": 1.0, "AUU": 0.75, "AUA": 0.35,
        "CUG": 1.0, "CUC": 0.48, "GUG": 0.95, "UUG": 0.30, "CUU": 0.30, "UUA": 0.15, "CUA": 0.15,
        "AAG": 1.0, "AAA": 0.75, "AUG": 1.0, "UUC": 1.0, "UUU": 0.80,
        "CCC": 1.0, "CCU": 0.85, "CCA": 0.80, "CCG": 0.35,
        "AGC": 1.0, "UCC": 0.90, "UCU": 0.75, "AGU": 0.60, "UCA": 0.55, "UCG": 0.20,
        "ACC": 1.0, "ACA": 0.75, "ACU": 0.65, "ACG": 0.30,
        "UGG": 1.0, "UAC": 1.0, "UAU": 0.75,
        "GUG": 1.0, "GUC": 0.50, "GUU": 0.35, "GUA": 0.22
    }
}

# =====================================================================
# 2. 常用限制酶辨識位點 (MCS + Type IIS Golden Gate)
# =====================================================================
RESTRICTION_SITES = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "NdeI": "CATATG",
    "XhoI": "CTCGAG",
    "NotI": "GCGGCCGC",
    "NcoI": "CCATGG",
    "SalI": "GTCGAC",
    "BsaI": "GGTCTC",    # Type IIS (Golden Gate)
    "BsmBI": "CGTCTC"    # Type IIS (Golden Gate)
}

# =====================================================================
# 3. 常用胜肽標籤與蛋白酶切位點
# =====================================================================
PEPTIDE_TAGS = {
    "6xHis": "HHHHHH",
    "FLAG": "DYKDDDDK",
    "HA": "YPYDVPDYA",
    "c-Myc": "EQKLISEEDL",
    "Strep-II": "WSHPQFEK",
    "GST": "MSPILGYWKIKGLVQPTRLLLEYLEEKYEEHLYERDEGDKWRNKKFELGLEFPNLPYYIDGDVKLTQSMAIIRYIADKHNMLGGCPKERAEISMLEGAVLDIRYGVSRIAYSKDFETLKVDFLSKLPEMLKMFEDRLCHKTYLNGDHVTHPDFMLYDALDVVLYMDPMCLDAFPKLVCFKKRIEAIPQIDKYLKSSKYIAWPLQGWQATFGGGDHPPK"
}

PROTEASE_SITES = {
    "None": "",
    "TEV protease (ENLYFQ↓G)": "ENLYFQG",
    "Thrombin (LVPR↓GS)": "LVPRGS",
    "PreScission / HRV 3C (LEVLFQ↓GP)": "LEVLFQGP",
    "Enterokinase (DDDDK↓)": "DDDDK"
}


# =====================================================================
# 4. 輔助序列轉換函式
# =====================================================================
def reverse_complement(dna_seq):
    comp = str.maketrans("ACGTURYKMSWBDHVN", "TGCAAYRMKSWVHDBN")
    return dna_seq.upper().translate(comp)[::-1]


def calc_primer_tm(seq):
    """
    計算寡核苷酸退火溫度 (Tm, °C)。短引子採 Wallace 法，長引子採經驗百分比法。
    """
    s = seq.upper()
    n = len(s)
    if n == 0:
        return 0.0
    gc = s.count("G") + s.count("C")
    if n < 14:
        return round(2.0 * (n - gc) + 4.0 * gc, 1)
    return round(64.9 + 41.0 * (gc - 16.4) / n, 1)


def reverse_translate_sequence(aa_seq, strategy="ecoli", add_start_stop=False):
    table_map = {
        "ecoli": ECOLI_RNA_CODONS,
        "human": HUMAN_RNA_CODONS,
        "deg": DEGENERATE_RNA_CODONS
    }
    table = table_map.get(strategy, ECOLI_RNA_CODONS)
    codons = [table.get(aa, "NNN") for aa in aa_seq.upper()]
    if add_start_stop:
        if not aa_seq.startswith("M"):
            codons.insert(0, "AUG")
        codons.append("UAA")
    mrna = "".join(codons)
    cdna = mrna.replace("U", "T")
    return mrna, cdna


def scan_restriction_sites(cdna_seq):
    hits = {}
    zero_cutters = []
    for enz, site in RESTRICTION_SITES.items():
        positions = []
        start = 0
        while True:
            idx = cdna_seq.find(site, start)
            if idx == -1:
                break
            positions.append(idx + 1)
            start = idx + 1
        if positions:
            hits[enz] = positions
        else:
            zero_cutters.append(enz)
    return hits, zero_cutters


def compute_cai_and_5prime_dg(mrna_seq, strategy="ecoli"):
    host = "human" if strategy == "human" else "ecoli"
    weights_tbl = CODON_WEIGHTS[host]
    codons = [mrna_seq[i:i+3] for i in range(0, len(mrna_seq) - 2, 3)]
    valid_w = [weights_tbl[c] for c in codons if c in weights_tbl]
    cai = float(np.exp(np.mean(np.log(valid_w)))) if valid_w else 0.0

    head_30 = mrna_seq[:30]
    gc_head = head_30.count("G") + head_30.count("C")
    au_head = head_30.count("A") + head_30.count("U")
    est_5prime_dg = round(-0.55 * gc_head - 0.15 * au_head + 4.2, 2)
    return cai, est_5prime_dg


# =====================================================================
# 5. 稀有密碼子簇掃描 (Rare Codon Cluster Scanner)
# =====================================================================
def scan_rare_codon_clusters(mrna_seq, host="ecoli"):
    """
    偵測連續出現的低豐度稀有密碼子 (RSCU 權重 < 0.15)。
    若 5 個密碼子視窗內出現 >= 2 個稀有密碼子，容易引發核糖體停頓 (Ribosome Stalling)。
    """
    weights = CODON_WEIGHTS.get(host, CODON_WEIGHTS["ecoli"])
    codons = [mrna_seq[i:i+3] for i in range(0, len(mrna_seq) - 2, 3)]
    rare_threshold = 0.15
    rare_indices = [idx for idx, c in enumerate(codons) if weights.get(c, 1.0) <= rare_threshold]

    clusters = []
    i = 0
    while i < len(rare_indices):
        curr_cluster = [rare_indices[i]]
        j = i + 1
        while j < len(rare_indices) and (rare_indices[j] - curr_cluster[-1]) <= 4:
            curr_cluster.append(rare_indices[j])
            j += 1
        if len(curr_cluster) >= 2:
            clusters.append({
                "start_codon": curr_cluster[0] + 1,
                "end_codon": curr_cluster[-1] + 1,
                "rare_codons": [f"{codons[k]} (#{k+1})" for k in curr_cluster],
                "count": len(curr_cluster)
            })
            i = j
        else:
            i += 1

    return {
        "rare_indices": rare_indices,
        "total_rare": len(rare_indices),
        "clusters": clusters
    }


# =====================================================================
# 6. 融合標籤與蛋白酶切位自動建構
# =====================================================================
def build_engineered_construct(
    core_aa,
    n_tag="None",
    c_tag="6xHis",
    protease="TEV protease (ENLYFQ↓G)",
    linker="GGGGS"
):
    """
    將目標蛋白序列依據 N 端標籤、蛋白酶切位、彈性 Linker 與 C 端標籤進行重組拼接。
    """
    n_seq = ""
    c_seq = ""
    feature_map = []

    pos = 1
    # 1. N-terminal Tag
    if n_tag != "None" and n_tag in PEPTIDE_TAGS:
        tag_seq = PEPTIDE_TAGS[n_tag]
        n_seq += tag_seq
        feature_map.append({"name": f"N-term {n_tag}", "start": pos, "end": pos + len(tag_seq) - 1, "type": "tag"})
        pos += len(tag_seq)

        # N-term Protease site
        if protease != "None" and protease in PROTEASE_SITES and PROTEASE_SITES[protease]:
            site_seq = PROTEASE_SITES[protease]
            n_seq += site_seq
            feature_map.append({"name": f"Cleavage Site ({protease.split()[0]})", "start": pos, "end": pos + len(site_seq) - 1, "type": "protease"})
            pos += len(site_seq)

        # Linker
        if linker:
            n_seq += linker
            feature_map.append({"name": "Flexible Linker", "start": pos, "end": pos + len(linker) - 1, "type": "linker"})
            pos += len(linker)

    # 2. Target Core Protein
    core_start = pos
    core_end = pos + len(core_aa) - 1
    feature_map.append({"name": "Target Protein Core", "start": core_start, "end": core_end, "type": "core"})
    pos += len(core_aa)

    # 3. C-terminal Tag
    if c_tag != "None" and c_tag in PEPTIDE_TAGS:
        if linker:
            c_seq += linker
            feature_map.append({"name": "Flexible Linker", "start": pos, "end": pos + len(linker) - 1, "type": "linker"})
            pos += len(linker)

        # C-term Protease site (if chosen and no N-tag)
        if n_tag == "None" and protease != "None" and protease in PROTEASE_SITES and PROTEASE_SITES[protease]:
            site_seq = PROTEASE_SITES[protease]
            c_seq += site_seq
            feature_map.append({"name": f"Cleavage Site ({protease.split()[0]})", "start": pos, "end": pos + len(site_seq) - 1, "type": "protease"})
            pos += len(site_seq)

        tag_seq = PEPTIDE_TAGS[c_tag]
        c_seq += tag_seq
        feature_map.append({"name": f"C-term {c_tag}", "start": pos, "end": pos + len(tag_seq) - 1, "type": "tag"})
        pos += len(tag_seq)

    full_aa = f"{n_seq}{core_aa}{c_seq}"
    return {
        "full_aa": full_aa,
        "n_tag": n_tag,
        "c_tag": c_tag,
        "protease": protease,
        "features": feature_map
    }


# =====================================================================
# 7. PCR 引子設計器：Sticky-End / Gibson / Golden Gate
# =====================================================================
def design_sticky_end_primers(cdna_seq, fwd_enz, rev_enz, anneal_len=21):
    clamp_5 = "GGAATT"
    fwd_site = RESTRICTION_SITES[fwd_enz]
    rev_site = RESTRICTION_SITES[rev_enz]

    fwd_anneal = cdna_seq[:anneal_len]
    fwd_full = f"{clamp_5}{fwd_site}{fwd_anneal}"

    core_3prime = cdna_seq[:-3] if cdna_seq.endswith(("TAA", "TAG", "TGA")) else cdna_seq
    rev_anneal = reverse_complement(core_3prime[-anneal_len:])
    rev_full = f"{clamp_5}{rev_site}{rev_anneal}"

    fwd_tm = calc_primer_tm(fwd_anneal)
    rev_tm = calc_primer_tm(rev_anneal)
    ta_rec = round(min(fwd_tm, rev_tm) - 3.0, 1)

    return {
        "method": "Sticky-End PCR (Restriction Enzyme)",
        "fwd_full": fwd_full,
        "fwd_anneal": fwd_anneal,
        "fwd_tm": fwd_tm,
        "fwd_len": len(fwd_full),
        "rev_full": rev_full,
        "rev_anneal": rev_anneal,
        "rev_tm": rev_tm,
        "rev_len": len(rev_full),
        "ta_rec": ta_rec,
        "fwd_enz": fwd_enz,
        "rev_enz": rev_enz
    }


def design_gibson_primers(cdna_seq, vec_5_overlap="CGCCATGGGGATCC", vec_3_overlap="CTCGAGTCTAGA", anneal_len=21):
    """
    設計 Gibson Assembly / In-Fusion 同源重組引子 (5' Overhang + 3' Anneal)
    """
    fwd_anneal = cdna_seq[:anneal_len]
    fwd_full = f"{vec_5_overlap.upper()}{fwd_anneal}"

    core_3prime = cdna_seq[:-3] if cdna_seq.endswith(("TAA", "TAG", "TGA")) else cdna_seq
    rev_anneal = reverse_complement(core_3prime[-anneal_len:])
    rev_full = f"{reverse_complement(vec_3_overlap.upper())}{rev_anneal}"

    fwd_tm_anneal = calc_primer_tm(fwd_anneal)
    rev_tm_anneal = calc_primer_tm(rev_anneal)
    fwd_tm_full = calc_primer_tm(fwd_full)
    rev_tm_full = calc_primer_tm(rev_full)
    ta_rec = round(min(fwd_tm_anneal, rev_tm_anneal) - 2.5, 1)

    return {
        "method": "Gibson Assembly / In-Fusion",
        "fwd_full": fwd_full,
        "fwd_anneal": fwd_anneal,
        "fwd_overlap": vec_5_overlap.upper(),
        "fwd_tm": fwd_tm_anneal,
        "fwd_tm_full": fwd_tm_full,
        "fwd_len": len(fwd_full),
        "rev_full": rev_full,
        "rev_anneal": rev_anneal,
        "rev_overlap": reverse_complement(vec_3_overlap.upper()),
        "rev_tm": rev_tm_anneal,
        "rev_tm_full": rev_tm_full,
        "rev_len": len(rev_full),
        "ta_rec": ta_rec
    }


def design_golden_gate_primers(cdna_seq, enzyme="BsaI", overhang_5="AATG", overhang_3="GCTT", anneal_len=21):
    """
    設計 Type IIS 限制酶 (BsaI 或 BsmBI) Golden Gate 組裝引子
    前綴結構：5' 保護鹼基 (4 nt) + 辨識序列 (GGTCTC) + 間隔鹼基 (N) + 黏性末端 (4 nt) + 退火序列
    """
    clamp = "TGCA"
    site = RESTRICTION_SITES[enzyme]  # BsaI: GGTCTC, BsmBI: CGTCTC
    spacer = "A"

    fwd_anneal = cdna_seq[:anneal_len]
    fwd_full = f"{clamp}{site}{spacer}{overhang_5.upper()}{fwd_anneal}"

    core_3prime = cdna_seq[:-3] if cdna_seq.endswith(("TAA", "TAG", "TGA")) else cdna_seq
    rev_anneal = reverse_complement(core_3prime[-anneal_len:])
    # 反向引子辨識位點需反向互補，以確保切除外側
    rev_site_rc = reverse_complement(site)
    rev_full = f"{clamp}{site}{spacer}{reverse_complement(overhang_3.upper())}{rev_anneal}"

    fwd_tm = calc_primer_tm(fwd_anneal)
    rev_tm = calc_primer_tm(rev_anneal)
    ta_rec = round(min(fwd_tm, rev_tm) - 3.0, 1)

    return {
        "method": f"Golden Gate Assembly ({enzyme})",
        "fwd_full": fwd_full,
        "fwd_anneal": fwd_anneal,
        "fwd_overhang": overhang_5.upper(),
        "fwd_tm": fwd_tm,
        "fwd_len": len(fwd_full),
        "rev_full": rev_full,
        "rev_anneal": rev_anneal,
        "rev_overhang": overhang_3.upper(),
        "rev_tm": rev_tm,
        "rev_len": len(rev_full),
        "ta_rec": ta_rec,
        "enzyme": enzyme
    }


# =====================================================================
# 8. GenBank (.gb) 標準質體圖譜檔案產生器
# =====================================================================
def generate_genbank_record(locus_name, cdna_seq, aa_seq, features=None, description="Recombinant Protein Construct"):
    """
    輸出相容於 SnapGene / Benchling / Geneious 的標準 GenBank (.gb) 文本格式
    """
    date_str = datetime.now().strftime("%d-%b-%Y").upper()
    dna_len = len(cdna_seq)

    lines = []
    lines.append(f"LOCUS       {locus_name[:16]:<16} {dna_len:>6} bp    DNA     linear   SYN {date_str}")
    lines.append(f"DEFINITION  {description}")
    lines.append(f"ACCESSION   {locus_name[:16]}")
    lines.append(f"VERSION     {locus_name[:16]}.1")
    lines.append("KEYWORDS    Protein Thermodynamics; Recombinant Expression.")
    lines.append("SOURCE      Synthetic DNA construct")
    lines.append("  ORGANISM  Synthetic construct")
    lines.append("            other sequences; artificial sequences.")
    lines.append("FEATURES             Location/Qualifiers")
    lines.append(f"     source          1..{dna_len}")
    lines.append("                     /organism=\"Synthetic construct\"")
    lines.append("                     /mol_type=\"other DNA\"")

    # CDS Feature
    lines.append(f"     CDS             1..{dna_len}")
    lines.append(f"                     /gene=\"{locus_name}\"")
    lines.append("                     /codon_start=1")
    lines.append(f"                     /translation=\"{aa_seq}\"")

    # Custom Features (Tags, Protease sites, etc.)
    if features:
        for f in features:
            f_start = (f["start"] - 1) * 3 + 1
            f_end = min(dna_len, f["end"] * 3)
            f_type = "misc_feature" if f.get("type") in ("tag", "protease", "linker") else "CDS"
            lines.append(f"     {f_type:<15} {f_start}..{f_end}")
            lines.append(f"                     /label=\"{f['name']}\"")
            lines.append(f"                     /note=\"Type: {f.get('type', 'feature')}\"")

    lines.append("ORIGIN")
    for i in range(0, dna_len, 60):
        chunk = cdna_seq[i:i+60].lower()
        sub_chunks = [chunk[j:j+10] for j in range(0, len(chunk), 10)]
        line_num = str(i + 1).rjust(9)
        lines.append(f"{line_num} {' '.join(sub_chunks)}")
    lines.append("//")

    return "\n".join(lines)


# =====================================================================
# 9. 蛋白質序列生化特性分析 (MW, pI, GRAVY, Charge)
# =====================================================================
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

AA_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15
}


def analyze_fasta_biochem(seq):
    s = [aa for aa in seq.upper() if aa in AA_MASS]
    L = len(s)
    if L == 0:
        return None

    mw_kda = (sum(AA_MASS[aa] for aa in s) - (L - 1) * 18.015) / 1000.0
    gravy = sum(KD_HYDROPATHY.get(aa, 0.0) for aa in s) / L
    arom_pct = sum(1 for aa in s if aa in ("F", "W", "Y")) / L * 100.0

    pos_cnt = sum(1 for aa in s if aa in ("K", "R"))
    neg_cnt = sum(1 for aa in s if aa in ("D", "E"))
    his_cnt = s.count("H")
    cys_cnt = s.count("C")
    tyr_cnt = s.count("Y")

    low, high = 0.0, 14.0
    for _ in range(40):
        mid = (low + high) / 2.0
        pos_q = (
            1.0 / (1.0 + 10 ** (mid - 9.6))
            + pos_cnt / (1.0 + 10 ** (mid - 10.5))
            + his_cnt / (1.0 + 10 ** (mid - 6.0))
        )
        neg_q = (
            1.0 / (1.0 + 10 ** (3.6 - mid))
            + neg_cnt / (1.0 + 10 ** (4.2 - mid))
            + cys_cnt / (1.0 + 10 ** (8.3 - mid))
            + tyr_cnt / (1.0 + 10 ** (10.1 - mid))
        )
        if pos_q - neg_q > 0:
            low = mid
        else:
            high = mid
    pi_val = (low + high) / 2.0

    ph = 7.4
    net_charge = (
        1.0 / (1.0 + 10 ** (ph - 9.6))
        + pos_cnt / (1.0 + 10 ** (ph - 10.5))
        + his_cnt / (1.0 + 10 ** (ph - 6.0))
        - 1.0 / (1.0 + 10 ** (3.6 - ph))
        - neg_cnt / (1.0 + 10 ** (4.2 - ph))
        - cys_cnt / (1.0 + 10 ** (8.3 - ph))
        - tyr_cnt / (1.0 + 10 ** (10.1 - ph))
    )

    return {
        "mw_kda": mw_kda,
        "pi": pi_val,
        "gravy": gravy,
        "net_charge": net_charge,
        "arom_pct": arom_pct
    }

