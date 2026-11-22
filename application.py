import numpy as np
import streamlit as st
from data_loader import load_pdb
from thermodynamics import (
    calculate_thermodynamics,
    generate_energy_curve,
    simulate_temperature_sweep
)

from visualization import (
    create_energy_figure,
    create_funnel_figure,
    create_ca_backbone_figure,
    create_contact_map_figure,
    create_ramachandran_figure,
    create_kinetics_sweep_figure
)

# 1. 頁面設定 (必須為第一個 st 指令)
st.set_page_config(
    page_title="Protein Thermodynamics Analyzer",
    page_icon="🧬",
    layout="wide"
)

# 隱藏所有標題旁的 Streamlit 預設錨點連結圖示
st.markdown(
    """
    <style>
    [data-testid="stHeaderActionElements"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# 多語系字典 (繁體中文 / English)
# =====================================
TEXTS = {
    "zh": {
        "arr_analysis_title": "📊 阿瑞尼士與熱動力學自動判讀",
        "arr_tm_comp": "**熱穩定度對比（$T_m$）：** Model A 的熔解溫度為 **{tm_a:.1f} K**，Model B 為 **{tm_b:.1f} K**，**{higher}** 的抗熱變性能力高出 **{diff:.1f} K（°C）**。",
        "arr_enthalpy": "**表觀活化焓（阿瑞尼士斜率來源）：** Model A 折疊活化焓 $\\Delta H^\\ddagger_{{\\text{{fold}}}} = {ha_f:.1f}$ kJ/mol（去折疊 $\\Delta H^\\ddagger_{{\\text{{unfold}}}} = {ha_u:.1f}$ kJ/mol）；Model B 折疊活化焓 $\\Delta H^\\ddagger_{{\\text{{fold}}}} = {hb_f:.1f}$ kJ/mol（去折疊 $\\Delta H^\\ddagger_{{\\text{{unfold}}}} = {hb_u:.1f}$ kJ/mol）。",
        "arr_non_arrhenius": "**非阿瑞尼士曲率解析：** 折疊過程需克服負的構形熵變（$\\Delta S^\\ddagger_{{\\text{{fold}}}} < 0$），隨溫度上升 $-T\\Delta S^\\ddagger$ 懲罰項增大，使高溫端 $k_{{\\text{{fold}}}}$ 成長趨緩，而去折疊速率 $k_{{\\text{{unfold}}}}$ 陡峭上升並於 $T_m$ 處超越折疊速率。",
        "arr_halflife": "**25°C（298.15 K）折疊半衰期（$t_{{1/2}}$）：** Model A 約為 **{hl_a}**，Model B 約為 **{hl_b}**。",
        "tab_kinetics": "🌡️ 阿瑞尼士溫度掃描 (Arrhenius Sweep)",
        "temp_range_label": "模擬溫度範圍 (Kelvin)",
        "tm_metric_a": "Model A 動力學交會點 (Tm)",
        "tm_metric_b": "Model B 動力學交會點 (Tm)",
        "tm_none": "超出掃描區間",
        "tab_rama": "📐 拉氏圖 (Ramachandran Plot)",
        "rama_alpha": "右手 α-螺旋區比例",
        "rama_beta": "β-折板區比例",
        "rama_other": "環區 / 其他構形",
        "title": "🧬 蛋白質熱力學分析儀",
        "subtitle": "互動式三態折疊自由能剖面與 3D 能量地景視覺化平台。",
        "sidebar_struct": "📁 蛋白質結構輸入",
        "upload_label": "上傳 PDB 結構檔（選填）",
        "upload_help": "上傳 .pdb 檔案以檢視胜肽鏈資訊並推算折疊能量尺度。",
        "sidebar_param": "⚙️ 熱力學參數設定",
        "unit_caption": "能量單位：kJ/mol",
        "input_mode_label": "參數調整模式",
        "mode_slider": "滑桿模式 (Slider)",
        "mode_number": "精確數值 (Number)",
        "model_a_title": "Model A 參數",
        "model_b_title": "Model B 參數",
        "u_label": "未折疊態 (Unfolded, U)",
        "ts_label": "過渡態 (Transition State, TS)",
        "n_label": "天然態 (Native, N)",
        "struct_summary": "結構摘要",
        "chains_metric": "偵測胜肽鏈數",
        "res_metric": "總胺基酸殘基數",
        "est_dg_metric": "經驗折疊 ΔG 參考尺度",
        "est_dg_help": "依據球狀蛋白經驗法則（約 -0.25 kJ/mol / 殘基）推算之穩定度參考值。",
        "view_seq": "檢視各鏈細節與胺基酸序列 (FASTA)",
        "tab_ca_3d": "🧬 3D Cα 骨架模型",
        "tab_fasta": "📜 胺基酸序列 (FASTA)",
        "select_pdb_model": "選擇結構模型 (Model)",
        "ca_color_mode": "骨架著色模式",
        "color_by_chain": "依胜肽鏈區分 (By Chain)",
        "color_by_seq": "依 N→C 端序列漸層 (N → C Terminus)",
        "show_ca_nodes": "顯示 Cα 原子節點",
        "tab_contact": "🗺️ 殘基接觸圖 (Contact Map)",
        "tab_nt": "🧬 推演核苷酸序列 (mRNA / cDNA)",
        "select_chain": "選擇胜肽鏈 (Chain)",
        "contact_mode": "顯示模式",
        "contact_binary": "二元接觸圖 (Binary Contact)",
        "contact_dist": "連續距離矩陣 (Distance Matrix)",
        "contact_cutoff": "Cα 接觸距離閾值 (Å)",
        "nt_strategy": "密碼子推演策略",
        "nt_opt": "單一常用密碼子 (Codon-Optimized)",
        "nt_deg": "IUPAC 簡併密碼子 (Degenerate)",
        "nt_include_stops": "包含起始 (AUG/ATG) 與終止密碼子",
        "nt_mrna_label": "推演 mRNA 序列 (5' → 3')",
        "nt_cdna_label": "推演編碼區 cDNA 序列 (5' → 3')",
        "parse_fail": "PDB 檔案解析失敗：",
        "landscape_header": "自由能地景視覺化",
        "tab_2d": "📈 2D 反應座標比較圖",
        "tab_3d": "🌋 3D 折疊能量漏斗",
        "topo_label": "能量地景拓撲類型",
        "topo_smooth": "平滑漏斗 (Smooth Funnel - 理想二態)",
        "topo_rugged": "粗糙漏斗 (Rugged Funnel - 能量挫折)",
        "topo_multi": "多盆地地景 (Multi-basin - 錯誤折疊陷阱)",
        "target_model": "檢視目標模型",
        "opt_compare_both": "雙模型並排比較 (Model A vs Model B)",
        "rough_amp": "粗糙度振幅 (kJ/mol)",
        "trap_depth": "局部陷阱深度 (kJ/mol)",
        "metrics_header": "熱力學指標對照與分析判讀",
        "dg_fold": "折疊自由能 ΔG",
        "ea_fold": "折疊活化能 ΔG‡",
        "ea_unfold": "去折疊活化能",
        "kinetic_header": "動力學速率與穩定度綜合結論 (298.15 K)",
        "interp_a_better": "**熱力學穩定度：** Model A 的折疊 ΔG 較低（差值 {diff:.1f} kJ/mol），天然態構形更穩定。",
        "interp_b_better": "**熱力學穩定度：** Model B 的折疊 ΔG 較低（差值 {diff:.1f} kJ/mol），天然態構形更穩定。",
        "interp_equal": "**熱力學穩定度：** Model A 與 Model B 具有相同的折疊自由能變化。",
        "interp_unfavored": "**熱力學穩定度：** 至少有一個模型的折疊 ΔG ≥ 0，在目前條件下自發性折疊在熱力學上不利。",
        "kinetic_a_faster": "**折疊動力學：** Model A 跨越的能障較低，25°C 下理論折疊速率約為 Model B 的 **{ratio:.1f} 倍**。",
        "kinetic_b_faster": "**折疊動力學：** Model B 跨越的能障較低，25°C 下理論折疊速率約為 Model A 的 **{ratio:.1f} 倍**。",
        "kinetic_equal": "**折疊動力學：** 兩模型具有相同的折疊活化能障，理論折疊速率相當。",
        "kinetic_barrierless": "**折疊動力學：** 存在無能障下坡折疊（ΔG‡ ≤ 0），反應速率受限於構形擴散而非活化能障。",
        "warn_ts_a": "Model A 參數異常：過渡態（TS）能量未同時高於未折疊態與天然態（退化為無能障折疊）。",
        "warn_ts_b": "Model B 參數異常：過渡態（TS）能量未同時高於未折疊態與天然態（退化為無能障折疊）。",
        "footer_note": "註：本工具為教學與概念驗證用之三態熱力學示意模型，非全原子分子動力學（MD）模擬。"
    },
    "en": {
        "arr_analysis_title": "📊 Automated Arrhenius & Thermal Stability Analysis",
        "arr_tm_comp": "**Thermal Stability ($T_m$):** Model A melts at **{tm_a:.1f} K** vs. Model B at **{tm_b:.1f} K**; **{higher}** exhibits higher thermostability by **{diff:.1f} K (°C)**.",
        "arr_enthalpy": "**Apparent Activation Enthalpy (Arrhenius Slope):** Model A $\\Delta H^\\ddagger_{{\\text{{fold}}}} = {ha_f:.1f}$ kJ/mol ($\\Delta H^\\ddagger_{{\\text{{unfold}}}} = {ha_u:.1f}$ kJ/mol); Model B $\\Delta H^\\ddagger_{{\\text{{fold}}}} = {hb_f:.1f}$ kJ/mol ($\\Delta H^\\ddagger_{{\\text{{unfold}}}} = {hb_u:.1f}$ kJ/mol).",
        "arr_non_arrhenius": "**Non-Arrhenius Behavior:** Because folding incurs a conformational entropy penalty ($\\Delta S^\\ddagger_{{\\text{{fold}}}} < 0$), elevated temperatures amplify $-T\\Delta S^\\ddagger$, flattening $k_{{\\text{{fold}}}}$ while accelerating $k_{{\\text{{unfold}}}}$ until thermal denaturation dominates above $T_m$.",
        "arr_halflife": "**Folding Half-Life ($t_{{1/2}}$) at 25°C (298.15 K):** Model A ≈ **{hl_a}** | Model B ≈ **{hl_b}**.",
        "tab_kinetics": "🌡️ Arrhenius / Eyring Sweep",
        "temp_range_label": "Temperature Range (Kelvin)",
        "tm_metric_a": "Model A Kinetic Crossover (Tm)",
        "tm_metric_b": "Model B Kinetic Crossover (Tm)",
        "tm_none": "Outside range",
        "tab_rama": "📐 Ramachandran Plot",
        "rama_alpha": "Right-handed α-Helix Region",
        "rama_beta": "β-Sheet Region",
        "rama_other": "Loop / Other Regions",
        "tab_contact": "🗺️ Contact Map",
        "tab_nt": "🧬 Reverse Translation (mRNA / cDNA)",
        "select_chain": "Select Polypeptide Chain",
        "contact_mode": "Map Display Mode",
        "contact_binary": "Binary Contact Map",
        "contact_dist": "Distance Matrix (Å)",
        "contact_cutoff": "Cα Contact Cutoff (Å)",
        "nt_strategy": "Codon Selection Strategy",
        "nt_opt": "Single Preferred Codon (Optimized)",
        "nt_deg": "IUPAC Degenerate Codons",
        "nt_include_stops": "Include Start (AUG/ATG) & Stop Codon",
        "nt_mrna_label": "Deduced mRNA Sequence (5' → 3')",
        "nt_cdna_label": "Deduced Coding cDNA Sequence (5' → 3')",
        "tab_ca_3d": "🧬 3D Cα Backbone Model",
        "tab_fasta": "📜 FASTA Sequences",
        "select_pdb_model": "Select Structure Model",
        "ca_color_mode": "Coloring Mode",
        "color_by_chain": "By Polypeptide Chain",
        "color_by_seq": "N → C Terminus Gradient",
        "show_ca_nodes": "Show Cα Atom Markers",
        "title": "🧬 Protein Thermodynamics Analyzer",
        "subtitle": "Interactive three-state folding free-energy profile and 3D energy landscape visualization.",
        "sidebar_struct": "📁 Structure Input",
        "upload_label": "Upload PDB File (Optional)",
        "upload_help": "Upload a .pdb structure to inspect chains and estimate folding energy scale.",
        "sidebar_param": "⚙️ Thermodynamic Parameters",
        "unit_caption": "Unit: kJ/mol",
        "input_mode_label": "Parameter Input Mode",
        "mode_slider": "Slider Mode",
        "mode_number": "Number Input",
        "model_a_title": "Model A Parameters",
        "model_b_title": "Model B Parameters",
        "u_label": "Unfolded State (U)",
        "ts_label": "Transition State (TS)",
        "n_label": "Native State (N)",
        "struct_summary": "Structure Summary",
        "chains_metric": "Detected Chains",
        "res_metric": "Total Amino Acid Residues",
        "est_dg_metric": "Empirical Folding ΔG Scale",
        "est_dg_help": "Rough empirical scale (~ -0.25 kJ/mol per residue) for comparison with Model A/B.",
        "view_seq": "View Chain Details & FASTA Sequences",
        "parse_fail": "PDB parsing failed:",
        "landscape_header": "Free Energy Landscape",
        "tab_2d": "📈 2D Reaction Coordinate",
        "tab_3d": "🌋 3D Folding Funnel",
        "topo_label": "Landscape Topology",
        "topo_smooth": "Smooth Funnel (Ideal)",
        "topo_rugged": "Rugged Funnel (Frustrated)",
        "topo_multi": "Multi-basin (Misfolding Traps)",
        "target_model": "Target Model",
        "opt_compare_both": "Compare Both (Model A vs Model B)",
        "rough_amp": "Ruggedness Amplitude (kJ/mol)",
        "trap_depth": "Trap Basin Depth (kJ/mol)",
        "metrics_header": "Thermodynamic Metrics & Interpretation",
        "dg_fold": "Folding ΔG",
        "ea_fold": "Folding Barrier ΔG‡",
        "ea_unfold": "Unfolding Barrier",
        "kinetic_header": "Thermodynamic & Kinetic Summary (298.15 K)",
        "interp_a_better": "**Thermodynamic Stability:** Model A has a more favorable folding ΔG (by {diff:.1f} kJ/mol), yielding a more stable native state.",
        "interp_b_better": "**Thermodynamic Stability:** Model B has a more favorable folding ΔG (by {diff:.1f} kJ/mol), yielding a more stable native state.",
        "interp_equal": "**Thermodynamic Stability:** Model A and Model B have identical folding free-energy changes.",
        "interp_unfavored": "**Thermodynamic Stability:** At least one model has ΔG ≥ 0; spontaneous folding is not thermodynamically favored.",
        "kinetic_a_faster": "**Folding Kinetics:** Model A faces a lower barrier, folding approximately **{ratio:.1f}× faster** than Model B at 25°C.",
        "kinetic_b_faster": "**Folding Kinetics:** Model B faces a lower barrier, folding approximately **{ratio:.1f}× faster** than Model A at 25°C.",
        "kinetic_equal": "**Folding Kinetics:** Both models share the same folding activation barrier and theoretical folding rate.",
        "kinetic_barrierless": "**Folding Kinetics:** Downhill/barrierless folding detected (ΔG‡ ≤ 0); rate is diffusion-limited.",
        "warn_ts_a": "Model A Notice: Transition state energy is not higher than both Unfolded and Native states (barrierless regime).",
        "warn_ts_b": "Model B Notice: Transition state energy is not higher than both Unfolded and Native states (barrierless regime).",
        "footer_note": "Note: Illustrative three-state thermodynamic model for educational visualization; not an all-atom molecular dynamics simulation."
    }
}

# =====================================
# 初始化 Session State
# =====================================
default_params = {
    "a_u": 0.0,
    "a_ts": 25.0,
    "a_n": -20.0,
    "b_u": 0.0,
    "b_ts": 18.0,
    "b_n": -12.0,
}
for k, v in default_params.items():
    if k not in st.session_state:
        st.session_state[k] = v


def render_param_control(label, state_key, min_val, max_val, use_slider):
    if use_slider:
        st.session_state[state_key] = st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[state_key]),
            step=1.0,
            key=f"slider_{state_key}"
        )
    else:
        st.session_state[state_key] = st.number_input(
            label,
            value=float(st.session_state[state_key]),
            step=1.0,
            key=f"num_{state_key}"
        )
    return st.session_state[state_key]

# =====================================
# 反向轉譯密碼子表 (Amino Acid -> RNA Codon)
# =====================================
PREFERRED_RNA_CODONS = {
    "A": "GCU", "R": "CGU", "N": "AAC", "D": "GAU", "C": "UGC",
    "Q": "CAG", "E": "GAA", "G": "GGU", "H": "CAC", "I": "AUU",
    "L": "CUG", "K": "AAA", "M": "AUG", "F": "UUU", "P": "CCG",
    "S": "UCU", "T": "ACC", "W": "UGG", "Y": "UAU", "V": "GUU",
    "*": "UAA", "X": "NNN"
}

DEGENERATE_RNA_CODONS = {
    "A": "GCN", "R": "CGN", "N": "AAY", "D": "GAY", "C": "UGY",
    "Q": "CAR", "E": "GAR", "G": "GGN", "H": "CAY", "I": "AUH",
    "L": "CUN", "K": "AAR", "M": "AUG", "F": "UUY", "P": "CCN",
    "S": "UCN", "T": "ACN", "W": "UGG", "Y": "UAY", "V": "GUN",
    "*": "UAR", "X": "NNN"
}


def reverse_translate_sequence(aa_seq, degenerate=False, add_start_stop=False):
    table = DEGENERATE_RNA_CODONS if degenerate else PREFERRED_RNA_CODONS
    codons = [table.get(aa, "NNN") for aa in aa_seq.upper()]
    if add_start_stop:
        if not aa_seq.startswith("M"):
            codons.insert(0, "AUG")
        codons.append("UAA")
    mrna = "".join(codons)
    cdna = mrna.replace("U", "T")
    return mrna, cdna

# =====================================
# 側邊欄：語言切換、PDB 上傳與雙模式參數調整
# =====================================
with st.sidebar:
    lang_choice = st.radio(
        "語言 / Language",
        ["繁體中文", "English"],
        horizontal=True
    )
    lang = "zh" if lang_choice == "繁體中文" else "en"
    t = TEXTS[lang]

    st.divider()
    st.subheader(t["sidebar_struct"])
    uploaded_pdb = st.file_uploader(
        t["upload_label"],
        type=["pdb"],
        help=t["upload_help"]
    )

    st.divider()
    st.subheader(t["sidebar_param"])
    st.caption(t["unit_caption"])

    input_mode = st.radio(
        t["input_mode_label"],
        [t["mode_slider"], t["mode_number"]],
        horizontal=True
    )
    use_slider = (input_mode == t["mode_slider"])

    with st.expander(t["model_a_title"], expanded=True):
        a_unfolded = render_param_control(t["u_label"], "a_u", -30.0, 30.0, use_slider)
        a_transition = render_param_control(t["ts_label"], "a_ts", 0.0, 60.0, use_slider)
        a_native = render_param_control(t["n_label"], "a_n", -60.0, 10.0, use_slider)

    with st.expander(t["model_b_title"], expanded=True):
        b_unfolded = render_param_control(t["u_label"], "b_u", -30.0, 30.0, use_slider)
        b_transition = render_param_control(t["ts_label"], "b_ts", 0.0, 60.0, use_slider)
        b_native = render_param_control(t["n_label"], "b_n", -60.0, 10.0, use_slider)

# =====================================
# 主畫面標題區
# =====================================
st.title(t["title"])
st.markdown(t["subtitle"])

# =====================================
# 區塊 1：PDB 結構摘要與 3D Cα 骨架模型
# =====================================
if uploaded_pdb is not None:
    try:
        results = load_pdb(uploaded_pdb)
        total_chains = len(results)
        total_residues = sum(item["residue_count"] for item in results)
        est_delta_g = -0.25 * total_residues

        st.subheader(f"{t['struct_summary']}: `{uploaded_pdb.name}`")
        m1, m2, m3 = st.columns(3)
        m1.metric(t["chains_metric"], f"{total_chains}")
        m2.metric(t["res_metric"], f"{total_residues} aa")
        m3.metric(
            t["est_dg_metric"],
            f"{est_delta_g:.1f} kJ/mol",
            help=t["est_dg_help"]
        )

        # 雙分頁：3D Cα 骨架模型 vs FASTA 序列
        subtab_3d, subtab_contact, subtab_rama, subtab_seq, subtab_nt = st.tabs([
            t["tab_ca_3d"],
            t["tab_contact"],
            t["tab_rama"],
            t["tab_fasta"],
            t["tab_nt"]
        ])

        with subtab_3d:
            available_models = sorted(list(set(item["model"] for item in results)))
            ca_col1, ca_col2, ca_col3 = st.columns([1.1, 1.4, 1.0])

            with ca_col1:
                chosen_pdb_model = st.selectbox(
                    t["select_pdb_model"],
                    available_models,
                    format_func=lambda m: f"Model {m}",
                    key="ca_model_select"
                )

            with ca_col2:
                ca_color = st.selectbox(
                    t["ca_color_mode"],
                    ["chain", "sequence"],
                    format_func=lambda x: t["color_by_chain"] if x == "chain" else t["color_by_seq"],
                    key="ca_color_select"
                )

            with ca_col3:
                st.write("")
                st.write("")
                show_nodes = st.checkbox(t["show_ca_nodes"], value=True, key="ca_nodes_chk")

            fig_ca = create_ca_backbone_figure(
                results,
                selected_model=chosen_pdb_model,
                color_mode=ca_color,
                show_markers=show_nodes
            )
            st.plotly_chart(fig_ca, use_container_width=True, key="pdb_ca_3d_chart")

        with subtab_contact:
            cm_col1, cm_col2, cm_col3 = st.columns([1.3, 1.3, 1.4])
            chain_options = list(range(len(results)))

            with cm_col1:
                chosen_chain_idx = st.selectbox(
                    t["select_chain"],
                    chain_options,
                    format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                    key="cm_chain_select"
                )

            with cm_col2:
                cm_mode = st.selectbox(
                    t["contact_mode"],
                    ["binary", "distance"],
                    format_func=lambda x: t["contact_binary"] if x == "binary" else t["contact_dist"],
                    key="cm_mode_select"
                )

            with cm_col3:
                cm_cutoff = st.slider(
                    t["contact_cutoff"],
                    min_value=4.0,
                    max_value=15.0,
                    value=8.0,
                    step=0.5,
                    disabled=(cm_mode == "distance"),
                    key="cm_cutoff_slider"
                )

            fig_cm = create_contact_map_figure(
                results[chosen_chain_idx],
                cutoff=cm_cutoff,
                map_mode=cm_mode
            )
            st.plotly_chart(fig_cm, use_container_width=True, key="pdb_contact_map_chart")

        with subtab_seq:
            for item in results:
                st.markdown(
                    f"**Model {item['model']} | Chain {item['chain']}** "
                    f"(`{item['residue_count']}` residues)"
                )
                st.code(item["sequence"], language=None)

        with subtab_nt:
            nt_col1, nt_col2 = st.columns([1.4, 1.6])
            with nt_col1:
                nt_strat = st.radio(
                    t["nt_strategy"],
                    ["opt", "deg"],
                    format_func=lambda x: t["nt_opt"] if x == "opt" else t["nt_deg"],
                    horizontal=True,
                    key="nt_strat_radio"
                )
            with nt_col2:
                st.write("")
                add_ends = st.checkbox(t["nt_include_stops"], value=False, key="nt_ends_chk")

            is_deg = (nt_strat == "deg")
            for item in results:
                mrna_seq, cdna_seq = reverse_translate_sequence(
                    item["sequence"],
                    degenerate=is_deg,
                    add_start_stop=add_ends
                )
                nt_len = len(mrna_seq)
                gc_count = mrna_seq.count("G") + mrna_seq.count("C")
                gc_pct = (gc_count / nt_len * 100.0) if nt_len > 0 else 0.0

                with st.expander(
                    f"Model {item['model']} | Chain {item['chain']} — {nt_len} nt (GC: {gc_pct:.1f}%)",
                    expanded=(len(results) == 1)
                ):
                    st.caption(t["nt_mrna_label"])
                    st.code(mrna_seq, language=None)
                    st.caption(t["nt_cdna_label"])
                    st.code(cdna_seq, language=None)
            
            with subtab_rama:
                rama_col1, rama_col2 = st.columns([1.4, 2.6])
                chain_options = list(range(len(results)))

                with rama_col1:
                    rama_chain_idx = st.selectbox(
                        t["select_chain"],
                        chain_options,
                        format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                        key="rama_chain_select"
                    )

                selected_rama_item = results[rama_chain_idx]
                r_list = selected_rama_item.get("rama_data", [])
                total_angles = len(r_list)
    
                # 統計二級結構區域殘基佔比
                with rama_col2:
                    if total_angles > 0:
                        alpha_cnt = sum(1 for d in r_list if -110 <= d["phi"] <= -35 and -70 <= d["psi"] <= -15)
                        beta_cnt = sum(1 for d in r_list if -165 <= d["phi"] <= -45 and 90 <= d["psi"] <= 175)
                        other_cnt = total_angles - alpha_cnt - beta_cnt
    
                        s1, s2, s3 = st.columns(3)
                        s1.metric(t["rama_alpha"], f"{alpha_cnt / total_angles * 100:.1f}%")
                        s2.metric(t["rama_beta"], f"{beta_cnt / total_angles * 100:.1f}%")
                        s3.metric(t["rama_other"], f"{other_cnt / total_angles * 100:.1f}%")

                fig_rama = create_ramachandran_figure(selected_rama_item)
                st.plotly_chart(fig_rama, use_container_width=True, key="pdb_rama_chart")
        st.divider()

    except Exception as e:
        st.error(f"{t['parse_fail']} {e}")

# =====================================
# 熱力學計算與資料封裝
# =====================================
model_a_results = calculate_thermodynamics(a_unfolded, a_transition, a_native)
model_b_results = calculate_thermodynamics(b_unfolded, b_transition, b_native)

a_x, a_y = generate_energy_curve(a_unfolded, a_transition, a_native)
b_x, b_y = generate_energy_curve(b_unfolded, b_transition, b_native)

model_a = {
    "x": a_x,
    "y": a_y,
    "g_unfolded": a_unfolded,
    "g_transition": a_transition,
    "g_native": a_native
}

model_b = {
    "x": b_x,
    "y": b_y,
    "g_unfolded": b_unfolded,
    "g_transition": b_transition,
    "g_native": b_native
}

# =====================================
# 區塊 2：能量地景視覺化
# =====================================
st.subheader(t["landscape_header"])

tab_2d, tab_3d, tab_kinetics = st.tabs([t["tab_2d"], t["tab_3d"], t["tab_kinetics"]])

with tab_2d:
    fig_2d = create_energy_figure(model_a, model_b)
    st.plotly_chart(fig_2d, use_container_width=True, key="energy_2d_chart")

with tab_3d:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.3, 1.3, 1.4])

    with ctrl_col1:
        funnel_type = st.selectbox(
            t["topo_label"],
            ["smooth", "rugged", "multi"],
            format_func=lambda x: {
                "smooth": t["topo_smooth"],
                "rugged": t["topo_rugged"],
                "multi": t["topo_multi"]
            }[x]
        )

    with ctrl_col2:
        target_choice = st.selectbox(
            t["target_model"],
            ["Model A", "Model B", "Compare Both"],
            format_func=lambda x: t["opt_compare_both"] if x == "Compare Both" else x
        )

    with ctrl_col3:
        if funnel_type == "rugged":
            roughness_amp = st.slider(
                t["rough_amp"],
                min_value=0.0,
                max_value=6.0,
                value=2.5,
                step=0.5
            )
            basin_depth = 9.0
        elif funnel_type == "multi":
            basin_depth = st.slider(
                t["trap_depth"],
                min_value=2.0,
                max_value=20.0,
                value=9.0,
                step=1.0
            )
            roughness_amp = 2.5
        else:
            roughness_amp = 0.0
            basin_depth = 9.0

    if target_choice == "Compare Both":
        # 計算統一的 Z 軸與色階範圍，確保兩座漏斗深度可直接視覺對照
        extra_down = basin_depth if funnel_type == "multi" else (roughness_amp if funnel_type == "rugged" else 0.0)
        extra_up = (
            max(0.0, a_transition - a_unfolded, b_transition - b_unfolded) * 0.15
            + (roughness_amp * 1.5 if funnel_type == "rugged" else 0.0)
        )
        shared_z_range = [
            min(a_native, b_native) - extra_down - 1.0,
            max(a_unfolded, b_unfolded) + extra_up + 1.0
        ]

        col_funnel_a, col_funnel_b = st.columns(2)
        with col_funnel_a:
            fig_3d_a = create_funnel_figure(
                model_a,
                funnel_type=funnel_type,
                roughness_amplitude=roughness_amp,
                basin_depth=basin_depth,
                z_range=shared_z_range,
                model_label="Model A",
                height=600
            )
            st.plotly_chart(fig_3d_a, use_container_width=True, key="funnel_3d_compare_a")

        with col_funnel_b:
            fig_3d_b = create_funnel_figure(
                model_b,
                funnel_type=funnel_type,
                roughness_amplitude=roughness_amp,
                basin_depth=basin_depth,
                z_range=shared_z_range,
                model_label="Model B",
                height=600
            )
            st.plotly_chart(fig_3d_b, use_container_width=True, key="funnel_3d_compare_b")
    else:
        selected_data = model_a if target_choice == "Model A" else model_b
        fig_3d = create_funnel_figure(
            selected_data,
            funnel_type=funnel_type,
            roughness_amplitude=roughness_amp,
            basin_depth=basin_depth,
            model_label=target_choice,
            height=720
        )
        st.plotly_chart(fig_3d, use_container_width=True, key="funnel_3d_chart")

with tab_kinetics:
    k_col1, k_col2, k_col3, k_col4 = st.columns([1.5, 1.3, 1.1, 1.1])

    with k_col1:
        t_min, t_max = st.slider(
            t["temp_range_label"],
            min_value=250.0,
            max_value=450.0,
            value=(275.0, 400.0),
            step=5.0,
            key="arrhenius_temp_slider"
        )

    with k_col2:
        ds_fold = st.slider(
            "ΔS_fold (kJ/mol·K)",
            min_value=-0.50,
            max_value=-0.05,
            value=-0.25,
            step=0.05,
            key="arrhenius_ds_slider"
        )

    sweep_a = simulate_temperature_sweep(
        model_a_results["delta_g_fold_barrier"],
        model_a_results["delta_g_unfold_barrier"],
        delta_s_fold=ds_fold,
        t_min=t_min,
        t_max=t_max
    )
    sweep_b = simulate_temperature_sweep(
        model_b_results["delta_g_fold_barrier"],
        model_b_results["delta_g_unfold_barrier"],
        delta_s_fold=ds_fold,
        t_min=t_min,
        t_max=t_max
    )

    with k_col3:
        tm_a_str = f"{sweep_a['t_m']:.1f} K ({sweep_a['t_m'] - 273.15:.1f} °C)" if sweep_a["t_m"] else t["tm_none"]
        st.metric(t["tm_metric_a"], tm_a_str)

    with k_col4:
        tm_b_str = f"{sweep_b['t_m']:.1f} K ({sweep_b['t_m'] - 273.15:.1f} °C)" if sweep_b["t_m"] else t["tm_none"]
        st.metric(t["tm_metric_b"], tm_b_str)

    fig_kinetics = create_kinetics_sweep_figure(sweep_a, sweep_b)
    st.plotly_chart(fig_kinetics, use_container_width=True, key="kinetics_sweep_chart")

# =====================================
    # 阿瑞尼士圖表即時數據判讀面板
    # =====================================
    with st.container(border=True):
        st.markdown(f"**{t['arr_analysis_title']}**")

        # 1. 計算表觀活化焓 ΔH‡ = ΔG‡(T0) + T0 * ΔS‡
        t_ref = 298.15
        ds_f_barrier = 0.6 * ds_fold
        ds_u_barrier = -0.4 * ds_fold

        ha_f = model_a_results["delta_g_fold_barrier"] + t_ref * ds_f_barrier
        ha_u = model_a_results["delta_g_unfold_barrier"] + t_ref * ds_u_barrier
        hb_f = model_b_results["delta_g_fold_barrier"] + t_ref * ds_f_barrier
        hb_u = model_b_results["delta_g_unfold_barrier"] + t_ref * ds_u_barrier

        # 2. 計算 25°C (298.15 K) 下的理論折疊速率與半衰期
        R = 8.314462618e-3
        prefactor_298 = 1e-6 * (1.380649e-23 * t_ref / 6.62607015e-34)
        kf_a_298 = prefactor_298 * np.exp(-np.clip(model_a_results["delta_g_fold_barrier"] / (R * t_ref), -100, 50))
        kf_b_298 = prefactor_298 * np.exp(-np.clip(model_b_results["delta_g_fold_barrier"] / (R * t_ref), -100, 50))

        def format_halflife(k_val):
            t_half = np.log(2.0) / k_val
            if t_half < 1e-3:
                return f"{t_half * 1e6:.2f} μs"
            elif t_half < 1.0:
                return f"{t_half * 1e3:.2f} ms"
            elif t_half < 60.0:
                return f"{t_half:.2f} s"
            else:
                return f"{t_half / 60.0:.2f} min"

        # 3. 輸出判讀條目
        tm_a_exact = (t_ref + model_a_results["delta_g_fold"] / ds_fold) if (ds_fold < 0 and model_a_results["delta_g_fold"] < 0) else None
        tm_b_exact = (t_ref + model_b_results["delta_g_fold"] / ds_fold) if (ds_fold < 0 and model_b_results["delta_g_fold"] < 0) else None

        if tm_a_exact is not None and tm_b_exact is not None and not np.isclose(tm_a_exact, tm_b_exact):
            higher_model = "Model A" if tm_a_exact > tm_b_exact else "Model B"
            st.markdown(
                t["arr_tm_comp"].format(
                    tm_a=tm_a_exact,
                    tm_b=tm_b_exact,
                    higher=higher_model,
                    diff=abs(tm_a_exact - tm_b_exact)
                )
            )

        st.markdown(
            t["arr_enthalpy"].format(
                ha_f=ha_f, ha_u=ha_u, hb_f=hb_f, hb_u=hb_u
            )
        )
        st.markdown(
            t["arr_halflife"].format(
                hl_a=format_halflife(kf_a_298),
                hl_b=format_halflife(kf_b_298)
            )
        )
        st.markdown(t["arr_non_arrhenius"])

# =====================================
# 區塊 3：熱力學指標對照與分析判讀 (無 delta 括號標示)
# =====================================
st.divider()
st.subheader(t["metrics_header"])

dg_a = model_a_results["delta_g_fold"]
dg_b = model_b_results["delta_g_fold"]
ea_a = model_a_results["delta_g_fold_barrier"]
ea_b = model_b_results["delta_g_fold_barrier"]
unfold_a = model_a_results["delta_g_unfold_barrier"]
unfold_b = model_b_results["delta_g_unfold_barrier"]

res_col_a, res_col_b = st.columns(2)

res_col_a, res_col_b = st.columns(2)

with res_col_a:
    with st.container(border=True):
        st.markdown("##### 🔵 Model A")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            t["dg_fold"],
            f"{dg_a:.1f} kJ/mol",
            delta=f"{dg_a - dg_b:+.1f} kJ/mol",
            delta_color="inverse"
        )
        c2.metric(
            t["ea_fold"],
            f"{ea_a:.1f} kJ/mol",
            delta=f"{ea_a - ea_b:+.1f} kJ/mol",
            delta_color="inverse"
        )
        c3.metric(
            t["ea_unfold"],
            f"{unfold_a:.1f} kJ/mol",
            delta=f"{unfold_a - unfold_b:+.1f} kJ/mol",
            delta_color="normal"
        )

with res_col_b:
    with st.container(border=True):
        st.markdown("##### 🟠 Model B")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            t["dg_fold"],
            f"{dg_b:.1f} kJ/mol",
            delta=f"{dg_b - dg_a:+.1f} kJ/mol",
            delta_color="inverse"
        )
        c2.metric(
            t["ea_fold"],
            f"{ea_b:.1f} kJ/mol",
            delta=f"{ea_b - ea_a:+.1f} kJ/mol",
            delta_color="inverse"
        )
        c3.metric(
            t["ea_unfold"],
            f"{unfold_b:.1f} kJ/mol",
            delta=f"{unfold_b - unfold_a:+.1f} kJ/mol",
            delta_color="normal"
        )
        
# 綜合結論面板
with st.container(border=True):
    st.markdown(f"**{t['kinetic_header']}**")

    if dg_a < 0 and dg_b < 0:
        if dg_a < dg_b:
            st.info(t["interp_a_better"].format(diff=abs(dg_a - dg_b)))
        elif dg_b < dg_a:
            st.info(t["interp_b_better"].format(diff=abs(dg_b - dg_a)))
        else:
            st.info(t["interp_equal"])
    else:
        st.warning(t["interp_unfavored"])

    RT = 8.314462618e-3 * 298.15
    if ea_a <= 0 or ea_b <= 0:
        st.write(t["kinetic_barrierless"])
    else:
        exponent = np.clip((ea_b - ea_a) / RT, -50.0, 50.0)
        rate_ratio = np.exp(exponent)
        if np.isclose(ea_a, ea_b):
            st.write(t["kinetic_equal"])
        elif ea_a < ea_b:
            st.write(t["kinetic_a_faster"].format(ratio=rate_ratio))
        else:
            st.write(t["kinetic_b_faster"].format(ratio=1.0 / rate_ratio))

    if not model_a_results["is_transition_maximum"]:
        st.warning(t["warn_ts_a"])
    if not model_b_results["is_transition_maximum"]:
        st.warning(t["warn_ts_b"])

st.caption(t["footer_note"])