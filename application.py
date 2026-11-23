import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from data_loader import (
    load_pdb,
    load_sample_ubiquitin,
    fetch_pdb_from_rcsb,
    analyze_disulfide_bonds,
    predict_mutation_ddg,
    analyze_hydrophobic_patches
)

from thermodynamics import (
    calculate_thermodynamics,
    generate_energy_curve,
    simulate_temperature_sweep,
    analyze_structural_biophysics,
    apply_mutation_to_thermodynamics
)

from visualization import (
    create_energy_figure,
    create_funnel_figure,
    create_ca_backbone_figure,
    create_contact_map_figure,
    create_ramachandran_figure,
    create_kinetics_sweep_figure,
    create_nucleotide_analysis_figure,
    create_3dmol_html,
    generate_standalone_report_html
)

from cloning_engine import (
    reverse_translate_sequence,
    scan_restriction_sites,
    compute_cai_and_5prime_dg,
    scan_rare_codon_clusters,
    build_engineered_construct,
    design_sticky_end_primers,
    design_gibson_primers,
    design_golden_gate_primers,
    generate_genbank_record,
    analyze_fasta_biochem,
    PEPTIDE_TAGS,
    PROTEASE_SITES
)

# =====================================
# 1. 頁面設定 (必須為第一個 st 指令)
# =====================================
st.set_page_config(
    page_title="Protein Thermodynamics & Recombinant Design Studio",
    page_icon="🧬",
    layout="wide"
)

# 注入現代質感 CSS：微調指標換行、按鈕懸停、容器自適應
st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow-wrap: break-word !important;
        line-height: 1.25 !important;
        font-size: clamp(0.78rem, 1.05vw, 0.95rem) !important;
    }
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow-wrap: break-word !important;
        font-size: clamp(1.15rem, 1.7vw, 1.6rem) !important;
        line-height: 1.2 !important;
    }
    .stButton > button p {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.3 !important;
    }
    button[data-baseweb="tab"] p {
        white-space: normal !important;
        font-size: clamp(0.8rem, 1vw, 0.95rem) !important;
        font-weight: 600 !important;
    }
    .block-container {
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        max-width: 100% !important;
    }
    [data-testid="stHeaderActionElements"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# 2. 多語系字典 (繁體中文 / English)
# =====================================
TEXTS = {
    "zh": {
        "title": "🧬 蛋白質熱力學與重組工程工作台",
        "subtitle": "整合 PDB 結構分析、三態折疊能量地景、突變穩定化預測、3Dmol 視覺化與重組分子選殖設計平台。",
        "sidebar_struct": "📁 蛋白質結構來源",
        "struct_input_method": "選擇結構輸入方式",
        "method_upload": "上傳本機 .pdb 檔案",
        "method_rcsb": "RCSB PDB 線上代碼輸入",
        "method_sample": "內建經典模型 (1UBQ)",
        "upload_label": "上傳 PDB 結構檔",
        "upload_help": "支援標準 PDB 格式檔案。",
        "rcsb_input_label": "輸入 4 碼 PDB ID (例: 1UBQ, 1CRN, 4HHB)",
        "btn_fetch_rcsb": "🌐 下載並分析結構",
        "preset_label": "快速測試熱門結構：",
        "btn_load_sample": "🧪 載入 Ubiquitin (1UBQ)",
        "btn_clear_sample": "清除結構",
        "sample_active_badge": "目前檢視：**Human Ubiquitin (PDB: 1UBQ, 76 aa)**",
        "sidebar_param": "⚙️ 熱力學能量模型設定",
        "unit_caption": "能量單位：kJ/mol",
        "input_mode_label": "參數調整模式",
        "mode_slider": "滑桿模式 (Slider)",
        "mode_number": "精確數值 (Number)",
        "model_a_title": "Model A (基準/野生型)",
        "model_b_title": "Model B (比較/突變型)",
        "u_label": "未折疊態 (Unfolded, U)",
        "ts_label": "過渡態 (Transition State, TS)",
        "n_label": "天然態 (Native, N)",
        "struct_summary": "結構摘要",
        "chains_metric": "偵測胜肽鏈數",
        "res_metric": "總胺基酸殘基數",
        "est_dg_metric": "經驗折疊 ΔG 參考尺度",
        "biophys_header": "🔬 結構拓撲與生物物理指標（驅動熱力學推算）",
        "metric_rco": "相對接觸序 (RCO)",
        "metric_rco_help": "衡量三級結構拓撲複雜度。RCO 越高，長程接觸越多，折疊活化能障（ΔG‡）越高。",
        "metric_lr": "長程接觸比例 (|i-j| ≥ 12)",
        "metric_rg": "迴轉半徑 (Rg)",
        "metric_core": "疏水核心埋藏率",
        "metric_core_help": "非極性殘基位於結構核心（< 0.75 Rg）的比例，反映疏水塌陷緊密度。",
        "btn_sync_model_a": "⚡ 將結構推算值套用至 Model A（TS = {ts:.1f}, N = {n:.1f} kJ/mol）",
        "tab_ca_3d": "🌟 3D 分子結構 (3Dmol / Ribbon)",
        "tab_contact": "🗺️ 殘基接觸與拓撲 (Contact Map)",
        "tab_rama": "🧭 拉氏圖與立體化學 (Ramachandran)",
        "tab_mutation": "🔬 蛋白質工程：突變與二硫鍵",
        "tab_fasta": "🧪 理化性質與 FASTA",
        "tab_nt": "🧬 基因密碼子與 mRNA 分析",
        "tab_cloning": "✂️ 分子選殖與引子設計",
        "tab_landscape": "🌋 熱力學地景與溫度動力學",
        "tab_export": "📥 報告匯出與 GenBank 下載",
        "select_chain": "選擇胜肽鏈 (Chain)",
        "cm_mode_label": "顯示模式",
        "cm_mode_binary": "二值化接觸圖 (Contact Map)",
        "cm_mode_dist": "連續距離熱圖 (Distance Matrix)",
        "cm_cutoff_label": "空間接觸閾值 (Cutoff, Å)",
        "ss_summary_title": "📊 二級結構組成與 SCOP 拓撲分類分析",
        "ss_helix_pct": "α-螺旋比例 (α-Helix)",
        "ss_sheet_pct": "β-折板比例 (β-Sheet)",
        "ss_coil_pct": "無規捲曲 / 環區 (Coil)",
        "ss_fold_class": "結構域拓撲類別",
        "class_all_alpha": "全 α-螺旋型 (All-α)",
        "class_all_beta": "全 β-折板型 (All-β)",
        "class_alpha_beta": "α/β 混合型 (α + β / α/β)",
        "class_coil": "無規捲曲主導型 (Disordered / Coil)",
        "rama_alpha": "右手 α-螺旋區比例",
        "rama_beta": "β-折板區比例",
        "rama_other": "環區 / 其他構形",
        "landscape_header": "自由能地景視覺化",
        "tab_2d": "📈 2D 反應座標比較圖",
        "tab_3d": "🌋 3D 折疊能量漏斗",
        "tab_kinetics": "🌡️ 阿瑞尼士溫度掃描 (Arrhenius Sweep)",
        "topo_label": "能量地景拓撲類型",
        "topo_smooth": "平滑漏斗 (Smooth Funnel - 理想二態)",
        "topo_rugged": "粗糙漏斗 (Rugged Funnel - 能量挫折)",
        "topo_multi": "多盆地地景 (Multi-basin - 錯誤折疊陷阱)",
        "target_model": "檢視目標模型",
        "opt_compare_both": "雙模型並排比較 (Model A vs Model B)",
        "rough_amp": "粗糙度振幅 (kJ/mol)",
        "trap_depth": "局部陷阱深度 (kJ/mol)",
        "temp_range_label": "模擬溫度範圍 (Kelvin)",
        "tm_metric_a": "Model A 熔解溫度 (Tm)",
        "tm_metric_b": "Model B 熔解溫度 (Tm)",
        "tm_none": "超出掃描區間",
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
        "warn_ts_a": "Model A 警告：過渡態（TS）能量未同時高於未折疊態與天然態。",
        "warn_ts_b": "Model B 警告：過渡態（TS）能量未同時高於未折疊態與天然態。",
        "footer_note": "註：本工具為結構生物物理與熱動力學輔助平台，提供 In Silico 預測與分子選殖實驗設計支援。"
    },
    "en": {
        "title": "🧬 Protein Thermodynamics & Recombinant Design Studio",
        "subtitle": "Integrated platform for PDB structural biophysics, folding energy landscapes, in silico mutation, 3Dmol visualization, and molecular cloning.",
        "sidebar_struct": "📁 Structure Source",
        "struct_input_method": "Select Structure Input",
        "method_upload": "Upload .pdb File",
        "method_rcsb": "Online RCSB PDB ID",
        "method_sample": "Built-in Sample (1UBQ)",
        "upload_label": "Upload PDB Structure File",
        "upload_help": "Standard PDB format supported.",
        "rcsb_input_label": "Enter 4-letter PDB ID (e.g. 1UBQ, 1CRN, 4HHB)",
        "btn_fetch_rcsb": "🌐 Fetch & Analyze",
        "preset_label": "Quick Presets:",
        "btn_load_sample": "🧪 Load Ubiquitin (1UBQ)",
        "btn_clear_sample": "Clear Structure",
        "sample_active_badge": "Active: **Human Ubiquitin (PDB: 1UBQ, 76 aa)**",
        "sidebar_param": "⚙️ Energy Model Parameters",
        "unit_caption": "Unit: kJ/mol",
        "input_mode_label": "Parameter Input Mode",
        "mode_slider": "Slider Mode",
        "mode_number": "Number Input",
        "model_a_title": "Model A (Wild-type / Ref)",
        "model_b_title": "Model B (Mutant / Comp)",
        "u_label": "Unfolded (U)",
        "ts_label": "Transition State (TS)",
        "n_label": "Native (N)",
        "struct_summary": "Structure Summary",
        "chains_metric": "Chains",
        "res_metric": "Total Residues",
        "est_dg_metric": "Empirical Folding ΔG Scale",
        "biophys_header": "🔬 Structural Biophysics & Topology Metrics",
        "metric_rco": "Relative Contact Order (RCO)",
        "metric_rco_help": "Topological complexity metric. Higher RCO correlates with higher folding barrier.",
        "metric_lr": "Long-Range Contacts (|i-j| ≥ 12)",
        "metric_rg": "Radius of Gyration (Rg)",
        "metric_core": "Hydrophobic Core Burial",
        "metric_core_help": "Fraction of nonpolar residues inside 0.75 Rg.",
        "btn_sync_model_a": "⚡ Apply Biophysics to Model A (TS = {ts:.1f}, N = {n:.1f} kJ/mol)",
        "tab_ca_3d": "🌟 3D Structure (3Dmol / Ribbon)",
        "tab_contact": "🗺️ Contact Map & Topology",
        "tab_rama": "🧭 Ramachandran & Stereochem",
        "tab_mutation": "🔬 Protein Engineering: Mutations & S-S",
        "tab_fasta": "🧪 Biochemistry & FASTA",
        "tab_nt": "🧬 Codon Opt & mRNA Analysis",
        "tab_cloning": "✂️ Cloning & Primer Designer",
        "tab_landscape": "🌋 Thermodynamics & Kinetics",
        "tab_export": "📥 Export Report & GenBank",
        "select_chain": "Select Chain",
        "cm_mode_label": "Display Mode",
        "cm_mode_binary": "Binary Contact Map",
        "cm_mode_dist": "Distance Matrix (Å)",
        "cm_cutoff_label": "Contact Distance Cutoff (Å)",
        "ss_summary_title": "📊 Secondary Structure Content & SCOP Fold Class",
        "ss_helix_pct": "α-Helix",
        "ss_sheet_pct": "β-Sheet",
        "ss_coil_pct": "Coil / Loop",
        "ss_fold_class": "SCOP Fold Class",
        "class_all_alpha": "All-α Protein",
        "class_all_beta": "All-β Protein",
        "class_alpha_beta": "α/β Mixed Fold",
        "class_coil": "Coil Dominated",
        "rama_alpha": "Right α-Helix Region",
        "rama_beta": "β-Sheet Region",
        "rama_other": "Loop / Other",
        "landscape_header": "Free Energy Landscapes",
        "tab_2d": "📈 2D Reaction Coordinate",
        "tab_3d": "🌋 3D Folding Funnel",
        "tab_kinetics": "🌡️ Arrhenius / Eyring Sweep",
        "topo_label": "Landscape Topology",
        "topo_smooth": "Smooth Funnel (Ideal)",
        "topo_rugged": "Rugged Funnel (Frustrated)",
        "topo_multi": "Multi-basin (Traps)",
        "target_model": "Target Model",
        "opt_compare_both": "Compare Both (Model A vs B)",
        "rough_amp": "Roughness Amplitude (kJ/mol)",
        "trap_depth": "Trap Depth (kJ/mol)",
        "temp_range_label": "Temperature Range (Kelvin)",
        "tm_metric_a": "Model A Melting Temp (Tm)",
        "tm_metric_b": "Model B Melting Temp (Tm)",
        "tm_none": "Outside Range",
        "metrics_header": "Thermodynamic Metrics & Analysis",
        "dg_fold": "Folding ΔG",
        "ea_fold": "Folding Barrier ΔG‡",
        "ea_unfold": "Unfolding Barrier",
        "kinetic_header": "Thermodynamic & Kinetic Summary (298.15 K)",
        "interp_a_better": "**Thermodynamic Stability:** Model A has a more favorable folding ΔG (by {diff:.1f} kJ/mol).",
        "interp_b_better": "**Thermodynamic Stability:** Model B has a more favorable folding ΔG (by {diff:.1f} kJ/mol).",
        "interp_equal": "**Thermodynamic Stability:** Models have identical folding free energies.",
        "interp_unfavored": "**Thermodynamic Stability:** At least one model has ΔG ≥ 0; folding is non-spontaneous.",
        "kinetic_a_faster": "**Folding Kinetics:** Model A folds approximately **{ratio:.1f}× faster** than Model B.",
        "kinetic_b_faster": "**Folding Kinetics:** Model B folds approximately **{ratio:.1f}× faster** than Model A.",
        "kinetic_equal": "**Folding Kinetics:** Both models have identical folding rates.",
        "kinetic_barrierless": "**Folding Kinetics:** Downhill folding detected.",
        "warn_ts_a": "Model A Notice: Transition state energy is not higher than both U and N.",
        "warn_ts_b": "Model B Notice: Transition state energy is not higher than both U and N.",
        "footer_note": "Note: In silico biophysics and thermodynamic design studio for academic & laboratory research."
    }
}

# =====================================
# 3. 狀態變數與回呼處理
# =====================================
default_params = {
    "a_u": 0.0, "a_ts": 25.0, "a_n": -20.0,
    "b_u": 0.0, "b_ts": 18.0, "b_n": -12.0,
}
for k, v in default_params.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_biophys_to_model_a(est_u, est_ts, est_n):
    st.session_state["a_u"] = est_u
    st.session_state["a_ts"] = est_ts
    st.session_state["a_n"] = est_n

def apply_mutation_to_model_b(ddg_kj):
    new_u, new_ts, new_n = apply_mutation_to_thermodynamics(
        st.session_state["a_u"],
        st.session_state["a_ts"],
        st.session_state["a_n"],
        ddg_kj
    )
    st.session_state["b_u"] = round(new_u, 1)
    st.session_state["b_ts"] = round(new_ts, 1)
    st.session_state["b_n"] = round(new_n, 1)

def render_param_control(label, state_key, min_val, max_val, use_slider):
    if use_slider:
        val = st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[state_key]),
            step=0.5,
            key=f"slider_{state_key}"
        )
    else:
        val = st.number_input(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[state_key]),
            step=0.5,
            key=f"num_{state_key}"
        )
    st.session_state[state_key] = val
    return val

# =====================================
# 4. 側邊欄配置
# =====================================
with st.sidebar:
    lang_choice = st.radio("語言 / Language", ["繁體中文(臺灣)", "English"], horizontal=True)
    lang = "zh" if lang_choice == "繁體中文(臺灣)" else "en"
    t = TEXTS[lang]

    st.divider()
    st.subheader(t["sidebar_struct"])

    input_mode = st.radio(
        t["struct_input_method"],
        [t["method_sample"], t["method_rcsb"], t["method_upload"]],
        index=0
    )

    if "active_results" not in st.session_state:
        st.session_state["active_results"] = None
        st.session_state["active_title"] = ""

    if input_mode == t["method_sample"]:
        if st.session_state["active_results"] is None or st.session_state["active_title"] != "1UBQ":
            st.session_state["active_results"] = load_sample_ubiquitin()
            st.session_state["active_title"] = "1UBQ (Ubiquitin)"

    elif input_mode == t["method_rcsb"]:
        col_rcsb, col_btn = st.columns([1.6, 1.0])
        with col_rcsb:
            user_pdb_id = st.text_input(t["rcsb_input_label"], value="1CRN", max_chars=4).strip().upper()
        with col_btn:
            st.write("")
            st.write("")
            if st.button(t["btn_fetch_rcsb"], use_container_width=True):
                with st.spinner(f"正在從 RCSB 下載 {user_pdb_id}..."):
                    try:
                        res = fetch_pdb_from_rcsb(user_pdb_id)
                        st.session_state["active_results"] = res
                        st.session_state["active_title"] = f"{user_pdb_id} (RCSB PDB)"
                        st.success(f"已成功載入 {user_pdb_id}！")
                    except Exception as err:
                        st.error(str(err))

        st.caption(t["preset_label"])
        preset_cols = st.columns(4)
        for idx, (p_id, p_name) in enumerate([("1UBQ", "Ubiquitin"), ("1CRN", "Crambin"), ("1L2Y", "Trp-cage"), ("4HHB", "Hemoglobin")]):
            if preset_cols[idx].button(p_id, key=f"pre_{p_id}", use_container_width=True):
                with st.spinner(f"載入 {p_id}..."):
                    try:
                        st.session_state["active_results"] = fetch_pdb_from_rcsb(p_id)
                        st.session_state["active_title"] = f"{p_id} ({p_name})"
                        st.rerun()
                    except Exception as err:
                        st.error(str(err))

    elif input_mode == t["method_upload"]:
        uploaded_file = st.file_uploader(t["upload_label"], type=["pdb"], help=t["upload_help"])
        if uploaded_file is not None:
            try:
                st.session_state["active_results"] = load_pdb(uploaded_file)
                st.session_state["active_title"] = uploaded_file.name
            except Exception as e:
                st.error(f"解析失敗: {e}")

    st.divider()
    st.subheader(t["sidebar_param"])
    st.caption(t["unit_caption"])

    param_input_style = st.radio(t["input_mode_label"], [t["mode_slider"], t["mode_number"]], horizontal=True)
    use_slider = (param_input_style == t["mode_slider"])

    with st.expander(t["model_a_title"], expanded=True):
        a_unfolded = render_param_control(t["u_label"], "a_u", -30.0, 30.0, use_slider)
        a_transition = render_param_control(t["ts_label"], "a_ts", 0.0, 60.0, use_slider)
        a_native = render_param_control(t["n_label"], "a_n", -60.0, 10.0, use_slider)

    with st.expander(t["model_b_title"], expanded=True):
        b_unfolded = render_param_control(t["u_label"], "b_u", -30.0, 30.0, use_slider)
        b_transition = render_param_control(t["ts_label"], "b_ts", 0.0, 60.0, use_slider)
        b_native = render_param_control(t["n_label"], "b_n", -60.0, 10.0, use_slider)

# =====================================
# 5. 主畫面標題與結構概況
# =====================================
st.title(t["title"])
st.markdown(t["subtitle"])

results = st.session_state.get("active_results")
struct_title = st.session_state.get("active_title", "Custom Protein")

if results is not None and len(results) > 0:
    total_chains = len(results)
    total_residues = sum(item["residue_count"] for item in results)
    est_delta_g = -0.25 * total_residues

    st.subheader(f"{t['struct_summary']}: `{struct_title}`")

    # 頂部快速指標列
    with st.container(border=True):
        bp_top1, bp_top2 = st.columns([2.5, 1.5])
        with bp_top1:
            st.markdown(f"**{t['biophys_header']}**")
        with bp_top2:
            bp_chain_idx = st.selectbox(
                "分析鏈",
                list(range(len(results))),
                format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                label_visibility="collapsed",
                key="top_bp_chain_selector"
            )

        target_bp_chain = results[bp_chain_idx]
        bp_stats = analyze_structural_biophysics(
            target_bp_chain["ca_coords"],
            target_bp_chain["sequence"]
        )

        if bp_stats is not None:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric(t["metric_rco"], f"{bp_stats['rco']:.3f}", help=t["metric_rco_help"])
            b2.metric(t["metric_lr"], f"{bp_stats['long_range_ratio'] * 100:.1f}%")
            b3.metric(t["metric_rg"], f"{bp_stats['rg']:.2f} Å")
            b4.metric(t["metric_core"], f"{bp_stats['core_burial_ratio'] * 100:.1f}%", help=t["metric_core_help"])

            st.button(
                t["btn_sync_model_a"].format(ts=bp_stats["est_ts"], n=bp_stats["est_n"]),
                use_container_width=True,
                key="btn_apply_biophys_a",
                on_click=apply_biophys_to_model_a,
                args=(bp_stats["est_u"], bp_stats["est_ts"], bp_stats["est_n"])
            )

    # =================================================================
    # 6. 核心分析功能多頁籤 (9 大專業功能 Hub)
    # =================================================================
    (
        tab_3d_view,
        tab_contact_view,
        tab_rama_view,
        tab_mut_view,
        tab_fasta_view,
        tab_nt_view,
        tab_cloning_view,
        tab_energy_view,
        tab_export_view
    ) = st.tabs([
        t["tab_ca_3d"],
        t["tab_contact"],
        t["tab_rama"],
        t["tab_mutation"],
        t["tab_fasta"],
        t["tab_nt"],
        t["tab_cloning"],
        t["tab_landscape"],
        t["tab_export"]
    ])

    # -------------------------------------------------------------
    # Tab 1: 3D 分子結構 (3Dmol.js WebGL & Plotly Ribbon)
    # -------------------------------------------------------------
    with tab_3d_view:
        view_mode_choice = st.radio(
            "檢視器模式",
            ["🌟 3Dmol.js WebGL 全分子模型 (Cartoon/Surface/Sticks/自轉)", "🎀 Plotly Cα 骨架飄帶 (Catmull-Rom / GC% 聯動)"],
            horizontal=True,
            key="3d_viewer_mode_radio"
        )

        if "3Dmol" in view_mode_choice:
            st.caption("💡 提示：使用滑鼠左鍵旋轉、右鍵平移、滾輪縮放；點擊或懸停於原子可查看殘基編號。頂部工具列可切換卡通、溶劑可及表面（SAS）或側鏈。")
            raw_pdb_string = target_bp_chain.get("raw_pdb", "")
            if not raw_pdb_string:
                raw_pdb_string = results[0].get("raw_pdb", "")
            html_3dmol = create_3dmol_html(raw_pdb_string, height=620)
            components.html(html_3dmol, height=640)
        else:
            ca_col1, ca_col2, ca_col3 = st.columns([1.0, 1.6, 1.0])
            available_models = sorted(list(set(item["model"] for item in results)))

            with ca_col1:
                chosen_pdb_model = st.selectbox(
                    "Model", available_models, format_func=lambda m: f"Model {m}", key="ca_model_select"
                )
            with ca_col2:
                ca_color = st.selectbox(
                    "著色模式",
                    ["ribbon", "chain", "sequence", "codon_gc"],
                    format_func=lambda x: {
                        "ribbon": "🎀 二級結構飄帶 (Helix/Sheet)",
                        "chain": "依胜肽鏈區分 (Chain)",
                        "sequence": "N → C 端序列漸層",
                        "codon_gc": "密碼子 GC% 聯動映射"
                    }[x],
                    key="ca_color_select"
                )
            with ca_col3:
                st.write("")
                st.write("")
                show_nodes = st.checkbox("顯示 Cα 原子點", value=(ca_color != "ribbon"), key="ca_nodes_chk")

            for item in results:
                _, c_dna = reverse_translate_sequence(item["sequence"], strategy="human", add_start_stop=False)
                item["codon_gc_list"] = [
                    round((c_dna[i*3:i*3+3].count("G") + c_dna[i*3:i*3+3].count("C")) / 3.0 * 100.0, 1)
                    for i in range(len(item["ca_coords"]))
                ]

            fig_ca = create_ca_backbone_figure(
                results,
                selected_model=chosen_pdb_model,
                color_mode=ca_color,
                show_markers=show_nodes,
                lang=lang
            )
            st.plotly_chart(fig_ca, use_container_width=True, key="pdb_ca_3d_chart")

        # SCOP 二級結構組成指標面板
        all_ss = target_bp_chain.get("ss_list", [])
        if len(all_ss) > 0:
            total_ss = len(all_ss)
            h_pct = all_ss.count("H") / total_ss * 100.0
            e_pct = all_ss.count("E") / total_ss * 100.0
            c_pct = all_ss.count("C") / total_ss * 100.0

            if h_pct >= 15.0 and e_pct >= 15.0:
                fold_class = t["class_alpha_beta"]
            elif h_pct >= 20.0 and e_pct < 15.0:
                fold_class = t["class_all_alpha"]
            elif e_pct >= 20.0 and h_pct < 15.0:
                fold_class = t["class_all_beta"]
            else:
                fold_class = t["class_coil"]

            with st.container(border=True):
                st.markdown(f"**{t['ss_summary_title']}**")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric(t["ss_helix_pct"], f"{h_pct:.1f}%")
                sc2.metric(t["ss_sheet_pct"], f"{e_pct:.1f}%")
                sc3.metric(t["ss_coil_pct"], f"{c_pct:.1f}%")
                sc4.metric(t["ss_fold_class"], fold_class)

    # -------------------------------------------------------------
    # Tab 2: 殘基接觸與拓撲 (Contact Map & Distance Matrix)
    # -------------------------------------------------------------
    with tab_contact_view:
        cm_col1, cm_col2, cm_col3 = st.columns([1.4, 1.3, 1.3])
        with cm_col1:
            cm_chain_idx = st.selectbox(
                t["select_chain"],
                list(range(len(results))),
                format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                key="cm_chain_select"
            )
        with cm_col2:
            cm_mode = st.selectbox(
                t["cm_mode_label"],
                ["binary", "distance"],
                format_func=lambda x: t["cm_mode_binary"] if x == "binary" else t["cm_mode_dist"],
                key="cm_mode_select"
            )
        with cm_col3:
            cm_cutoff = st.slider(
                t["cm_cutoff_label"],
                min_value=4.0, max_value=15.0, value=8.0, step=0.5,
                disabled=(cm_mode == "distance"),
                key="cm_cutoff_slider"
            )

        target_cm_item = results[cm_chain_idx]
        fig_cm = create_contact_map_figure(target_cm_item, cutoff=cm_cutoff, map_mode=cm_mode, lang=lang)
        st.plotly_chart(fig_cm, use_container_width=True, key="pdb_contact_map_chart")

        cm_coords = np.array(target_cm_item["ca_coords"], dtype=float)
        L_cm = len(cm_coords)
        if L_cm >= 4:
            diff_cm = cm_coords[:, np.newaxis, :] - cm_coords[np.newaxis, :, :]
            dist_cm = np.sqrt(np.sum(diff_cm**2, axis=-1))
            ii, jj = np.triu_indices(L_cm, k=3)
            c_mask = dist_cm[ii, jj] <= cm_cutoff
            seps = np.abs(jj[c_mask] - ii[c_mask])
            n_tot_c = len(seps)
            n_local = int(np.sum(seps < 12))
            n_long = int(np.sum(seps >= 12))
            long_pct = (n_long / n_tot_c * 100.0) if n_tot_c > 0 else 0.0

            with st.container(border=True):
                cm1, cm2, cm3 = st.columns(3)
                cm1.metric("總非相鄰接觸對" if lang == "zh" else "Non-local Contacts", f"{n_tot_c} ({n_tot_c / L_cm:.2f}/aa)")
                cm2.metric("局部接觸 (3 ≤ |i-j| < 12)", f"{n_local} ({100.0 - long_pct:.1f}%)")
                cm3.metric("長程三級接觸 (|i-j| ≥ 12)", f"{n_long} ({long_pct:.1f}%)")

                if long_pct >= 45.0:
                    st.info(f"📋 **拓撲診斷：** 長程三級接觸佔比達 **{long_pct:.1f}%**，具備高度協同折疊（Cooperative Folding）與緊密球狀核心。")
                else:
                    st.info(f"📋 **拓撲診斷：** 接觸訊號集中於對角線附近（局部接觸佔 **{100.0 - long_pct:.1f}%**），呈現伸展型或螺旋束拓撲。")

    # -------------------------------------------------------------
    # Tab 3: 拉氏圖與立體化學 (Ramachandran & Stereochemistry)
    # -------------------------------------------------------------
    with tab_rama_view:
        rama_col1, rama_col2 = st.columns([1.4, 2.6])
        with rama_col1:
            rama_chain_idx = st.selectbox(
                t["select_chain"],
                list(range(len(results))),
                format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                key="rama_chain_select"
            )

        target_rama_item = results[rama_chain_idx]
        r_list = target_rama_item.get("rama_data", [])
        total_angles = len(r_list)

        with rama_col2:
            if total_angles > 0:
                alpha_cnt = sum(1 for d in r_list if -110 <= d["phi"] <= -35 and -70 <= d["psi"] <= -15)
                beta_cnt = sum(1 for d in r_list if -165 <= d["phi"] <= -45 and 90 <= d["psi"] <= 175)
                other_cnt = total_angles - alpha_cnt - beta_cnt

                s1, s2, s3 = st.columns(3)
                s1.metric(t["rama_alpha"], f"{alpha_cnt / total_angles * 100:.1f}%")
                s2.metric(t["rama_beta"], f"{beta_cnt / total_angles * 100:.1f}%")
                s3.metric(t["rama_other"], f"{other_cnt / total_angles * 100:.1f}%")

        fig_rama = create_ramachandran_figure(target_rama_item, lang=lang)
        st.plotly_chart(fig_rama, use_container_width=True, key="pdb_rama_chart")

        if total_angles > 0:
            outliers = []
            favored_cnt = 0
            for idx_r, d in enumerate(r_list):
                phi = d.get("phi", 0.0)
                psi = d.get("psi", 0.0)
                res_name = d.get("res_name", "")
                res_num = d.get("res_num", idx_r + 1)
                in_alpha = (-140 <= phi <= -30) and (-80 <= psi <= 10)
                in_beta = (-180 <= phi <= -40) and (80 <= psi <= 180)
                in_left = (30 <= phi <= 100) and (0 <= psi <= 90)
                is_gly = res_name == "GLY"

                if in_alpha or in_beta or in_left or is_gly:
                    favored_cnt += 1
                else:
                    outliers.append(f"{res_name}{res_num} (φ={phi:.1f}°, ψ={psi:.1f}°)")

            fav_pct = favored_cnt / total_angles * 100.0
            with st.container(border=True):
                st.markdown(f"**🔬 主鏈立體化學品質診斷（核心允許區佔比：{fav_pct:.1f}%）**")
                if not outliers:
                    st.success("所有非甘胺酸殘基皆位於立體化學核心允許區，無主鏈空間位阻衝突。")
                else:
                    st.warning(f"偵測到 **{len(outliers)}** 個構象張力/異常殘基：`{', '.join(outliers[:8])}`" + (" ..." if len(outliers) > 8 else ""))

    # -------------------------------------------------------------
    # Tab 4: 蛋白質工程：突變與二硫鍵 (NEW!)
    # -------------------------------------------------------------
    with tab_mut_view:
        mut_chain_item = target_bp_chain

        # 4.1 In Silico 點突變與丙胺酸掃描
        with st.container(border=True):
            st.markdown("### 🧬 1. In Silico 點突變與丙胺酸掃描 (Alanine Scanning / ΔΔG 預測)")
            st.caption("預測特定殘基突變對熱力學穩定性 (ΔΔG) 與熔解溫度 (Tm) 的破壞或提升效果。")

            m_col1, m_col2, m_col3 = st.columns([1.5, 1.2, 1.3])
            res_options = mut_chain_item["res_nums"]
            with m_col1:
                sel_res_num = st.selectbox(
                    "選擇突變殘基位點",
                    res_options,
                    format_func=lambda n: f"殘基 #{n} ({mut_chain_item['sequence'][res_options.index(n)]}) - 區域: {mut_chain_item.get('ss_list', ['C']*len(res_options))[res_options.index(n)]}",
                    key="mut_res_selector"
                )
            with m_col2:
                amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
                sel_target_aa = st.selectbox("突變目標胺基酸", amino_acids, index=0, key="mut_target_aa_selector")

            mut_res = predict_mutation_ddg(mut_chain_item, sel_res_num, sel_target_aa)

            if mut_res is not None:
                with m_col3:
                    st.write("")
                    st.write("")
                    if st.button("⚡ 將突變體套用至 Model B", use_container_width=True, key="btn_apply_mut_b"):
                        apply_mutation_to_model_b(mut_res["ddg_kj"])
                        st.toast(f"已將突變體 {mut_res['mutation']} 參數同步至 Model B！可在熱力學地景中比對。")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("突變標記", mut_res["mutation"])
                mc2.metric("預測 ΔΔG (折疊自由能差)", f"{mut_res['ddg_kj']:+.1f} kJ/mol", delta=f"{mut_res['ddg_kcal']:+.2f} kcal/mol", delta_color="inverse")
                mc3.metric("預估 Tm 變化 (ΔTm)", f"{mut_res['delta_tm']:+.1f} °C", delta_color="normal")
                mc4.metric("穩定度影響評估", mut_res["impact"])

                if mut_res["is_buried"]:
                    st.info(f"💡 **環境診斷：** 殘基 #{sel_res_num} 位於疏水核心深處（周遭接觸數: {mut_res['local_contacts']}），若突變為較小側鏈將引發空洞效應（Cavity Penalty），導致顯著熱穩定性下降。")
                else:
                    st.info(f"💡 **環境診斷：** 殘基 #{sel_res_num} 暴露於蛋白質溶劑表面，突變主要受電荷、溶劑化及二級結構傾向性調控。")

        # 4.2 二硫鍵分析與全新工程推薦
        with st.container(border=True):
            st.markdown("### 🔗 2. 二硫鍵工程設計 (De Novo Disulfide Bond Engineering)")
            st.caption("依據 Hazes & Dijkstra 幾何法則（Cα-Cα 4.5–6.8 Å，Cβ-Cβ 3.4–4.8 Å），搜尋最佳雙半胱胺酸突變位點以鎖定結構並提高 Tm。")

            ss_analysis = analyze_disulfide_bonds(mut_chain_item)
            native_bonds = ss_analysis["native_bonds"]
            candidates = ss_analysis["candidates"]

            if native_bonds:
                st.markdown("**既存天然二硫鍵 (Native S-S):** " + " ｜ ".join([f"`CYS{b['res1']} ↔ CYS{b['res2']}` (SG-SG: {b['dist']:.2f} Å)" for b in native_bonds]))
            else:
                st.caption("本胜肽鏈內部未偵測到既有天然二硫鍵。")

            if candidates:
                st.markdown(f"**🌟 推薦全新工程二硫鍵位點（前 {len(candidates)} 佳候選對）：**")
                cols_head = st.columns([1.2, 1.2, 1.0, 1.0, 1.0, 1.2, 1.2])
                cols_head[0].markdown("**位點 1**")
                cols_head[1].markdown("**位點 2**")
                cols_head[2].markdown("**Cα 距離**")
                cols_head[3].markdown("**Cβ 距離**")
                cols_head[4].markdown("**序列間距**")
                cols_head[5].markdown("**預估 ΔΔG**")
                cols_head[6].markdown("**預期 Tm 提升**")

                for cand in candidates[:6]:
                    c_cols = st.columns([1.2, 1.2, 1.0, 1.0, 1.0, 1.2, 1.2])
                    c_cols[0].write(f"`{cand['res1_aa']}{cand['res1_num']} ({cand['res1_ss']})`")
                    c_cols[1].write(f"`{cand['res2_aa']}{cand['res2_num']} ({cand['res2_ss']})`")
                    c_cols[2].write(f"{cand['d_ca']:.2f} Å")
                    c_cols[3].write(f"{cand['d_cb']:.2f} Å")
                    c_cols[4].write(f"{cand['seq_dist']} aa")
                    c_cols[5].markdown(f":green[**{cand['est_ddg_kj']} kJ/mol**]")
                    c_cols[6].markdown(f":blue[**+{cand['est_tm_boost']} °C**]")
            else:
                st.info("目前結構幾何中未發現高度吻合二硫鍵理想幾何之殘基對。")

        # 4.3 表面疏水斑塊與聚集風險
        with st.container(border=True):
            st.markdown("### ⚠️ 3. 表面暴露疏水斑塊與聚集風險診斷 (Aggregation-Prone Patches)")
            patch_data = analyze_hydrophobic_patches(mut_chain_item)

            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("表面暴露疏水殘基數", f"{patch_data['exposed_count']} aa ({patch_data['exposed_pct']}%)")
            hc2.metric("高風險疏水叢集數", f"{patch_data['patches_count']} 個斑塊")
            hc3.metric("整體可溶性與聚集傾向", patch_data["risk_level"])

            if patch_data["patches"]:
                st.warning("⚠️ 偵測到表面疏水殘基聚集，重組表現時易形成包涵體或在純化過程發生自發性沉澱。建議考慮點突變為極性殘基（如 Ser/Thr）。")

    # -------------------------------------------------------------
    # Tab 5: 理化性質與 FASTA (Biochemistry & FASTA)
    # -------------------------------------------------------------
    with tab_fasta_view:
        for item in results:
            seq_str = item["sequence"]
            header_tag = f">Model_{item['model']}_Chain_{item['chain']} | {item['residue_count']} aa"
            fasta_block = f"{header_tag}\n{seq_str}"
            bio_stats = analyze_fasta_biochem(seq_str)

            with st.container(border=True):
                st.markdown(f"**Model {item['model']} | Chain {item['chain']}** (`{item['residue_count']}` residues)")
                if bio_stats is not None:
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("分子量 (MW)", f"{bio_stats['mw_kda']:.2f} kDa")
                    f2.metric("理論等電點 (pI)", f"{bio_stats['pi']:.2f}")
                    f3.metric("GRAVY 疏水性", f"{bio_stats['gravy']:+.3f}")
                    f4.metric("pH 7.4 淨電荷", f"{bio_stats['net_charge']:+.1f}")

                    sol_msg = "整體親水，重組表現可溶性良好" if bio_stats["gravy"] < 0 else "整體偏疏水（易形成包涵體）"
                    buf_msg = "帶負電（建議使用陰離子交換層析）" if bio_stats["net_charge"] < -1 else ("帶正電（建議陽離子交換層析）" if bio_stats["net_charge"] > 1 else "電中性（等電點沉澱風險高）")
                    st.caption(f"🧪 **生化診斷：** {sol_msg}；生理 pH 下{buf_msg}；芳香族殘基佔比 {bio_stats['arom_pct']:.1f}%。")

                st.code(fasta_block, language=None)

    # -------------------------------------------------------------
    # Tab 6: 基因密碼子與 mRNA 分析 (Codon Opt & mRNA ΔG)
    # -------------------------------------------------------------
    with tab_nt_view:
        nt_col1, nt_col2, nt_col3 = st.columns([1.4, 1.4, 1.2])
        with nt_col1:
            nt_chain_idx = st.selectbox(
                t["select_chain"],
                list(range(len(results))),
                format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                key="nt_chain_select"
            )
        with nt_col2:
            nt_strat = st.selectbox(
                "宿主密碼子優化策略",
                ["ecoli", "human", "deg"],
                format_func=lambda x: {"ecoli": "E. coli Codon-Optimized", "human": "Homo sapiens Optimized", "deg": "IUPAC 簡併密碼子"}[x],
                key="nt_strat_select"
            )
        with nt_col3:
            st.write("")
            st.write("")
            add_ends = st.checkbox("含起始 (ATG) 與終止碼", value=True, key="nt_ends_chk")

        target_nt_item = results[nt_chain_idx]
        mrna_seq, cdna_seq = reverse_translate_sequence(target_nt_item["sequence"], strategy=nt_strat, add_start_stop=add_ends)

        nt_len = len(cdna_seq)
        gc_total = (cdna_seq.count("G") + cdna_seq.count("C")) / nt_len * 100.0 if nt_len > 0 else 0.0
        wobble_bases = cdna_seq[2::3]
        gc3_pct = (wobble_bases.count("G") + wobble_bases.count("C")) / len(wobble_bases) * 100.0 if len(wobble_bases) > 0 else 0.0
        mw_kda = (nt_len * 617.96 + 36.04) / 1000.0

        nm1, nm2, nm3, nm4 = st.columns(4)
        nm1.metric("cDNA 全長", f"{nt_len} bp")
        nm2.metric("整體 GC 含量", f"{gc_total:.1f}%")
        nm3.metric("搖擺位 GC3%", f"{gc3_pct:.1f}%")
        nm4.metric("雙股 cDNA 分子量", f"{mw_kda:.2f} kDa")

        cai_val, dg_5prime = compute_cai_and_5prime_dg(mrna_seq, strategy=nt_strat)
        em1, em2 = st.columns(2)
        em1.metric("密碼子適應指數 (CAI)", f"{cai_val:.3f}")
        em2.metric("mRNA 5' 端前 30 nt 折疊 ΔG", f"{dg_5prime:.2f} kcal/mol")

        # 稀有密碼子簇掃描
        rare_report = scan_rare_codon_clusters(mrna_seq, host="ecoli" if nt_strat == "ecoli" else "human")
        if rare_report["clusters"]:
            st.warning(f"⚠️ **轉譯停頓警告：** 偵測到 {len(rare_report['clusters'])} 處低頻稀有密碼子簇，容易導致核糖體積聚與轉譯中斷！")
        else:
            st.success("✅ **密碼子順暢：** 未發現高風險稀有密碼子密集叢集，轉譯延伸預期順利。")

        fig_nt = create_nucleotide_analysis_figure(cdna_seq, window=15, lang=lang)
        st.plotly_chart(fig_nt, use_container_width=True, key="pdb_nt_analysis_chart")

        st.caption("推演編碼區 cDNA 序列 (5' → 3'):")
        st.code(cdna_seq, language=None)

    # -------------------------------------------------------------
    # Tab 7: 分子選殖與引子設計 (Cloning & Primers)
    # -------------------------------------------------------------
    with tab_cloning_view:
        st.markdown("### ✂️ 重組質體建構與 PCR 引子設計工作台")

        # 標籤與蛋白酶切位配置器
        with st.expander("🏷️ 融合標籤 (Tags) 與蛋白酶切位 (Protease Sites) 一鍵添加", expanded=True):
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                sel_n_tag = st.selectbox("N 端標籤 (N-term Tag)", ["None", "6xHis", "FLAG", "HA", "c-Myc", "Strep-II", "GST"], index=1)
            with tc2:
                sel_c_tag = st.selectbox("C 端標籤 (C-term Tag)", ["None", "6xHis", "FLAG", "HA", "c-Myc", "Strep-II"], index=0)
            with tc3:
                sel_protease = st.selectbox("蛋白酶切位", list(PROTEASE_SITES.keys()), index=1)
            with tc4:
                sel_linker = st.selectbox("彈性 Linker", ["GGGGS", "GGSGGS", "None"], index=0)

            linker_str = "" if sel_linker == "None" else sel_linker
            eng_construct = build_engineered_construct(
                target_bp_chain["sequence"],
                n_tag=sel_n_tag,
                c_tag=sel_c_tag,
                protease=sel_protease,
                linker=linker_str
            )
            st.caption(f"**重組全長構建體序列** ({len(eng_construct['full_aa'])} aa):")
            st.code(eng_construct["full_aa"], language=None)

        _, eng_cdna = reverse_translate_sequence(eng_construct["full_aa"], strategy="ecoli", add_start_stop=True)

        # 3 種選殖技術路徑切換
        cloning_method = st.radio(
            "選擇分子選殖策略",
            ["1. 限制酶黏性末端選殖 (Sticky-End / MCS)", "2. Gibson Assembly / In-Fusion 同源重組", "3. Type IIS (Golden Gate Assembly)"],
            horizontal=True
        )

        if "Sticky-End" in cloning_method:
            hits, zero_cutters = scan_restriction_sites(eng_cdna)
            with st.container(border=True):
                st.markdown("**質體 MCS 限制酶切位掃描結果：**")
                if hits:
                    hit_str = " ｜ ".join([f"**{enz}** (位置: {', '.join(map(str, pos))} bp)" for enz, pos in hits.items()])
                    st.warning(f"偵測到內部限制酶切位：{hit_str}")
                else:
                    st.success("序列內部無常見 MCS 限制酶切位，所有常用酶皆可安全使用。")
                st.caption(f"安全 0-cutter 酵素：{', '.join(zero_cutters)}")

                if len(zero_cutters) >= 2:
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        f_enz = st.selectbox("5' Forward 限制酶", zero_cutters, index=0)
                    with p2:
                        r_enz_opts = [e for e in zero_cutters if e != f_enz]
                        r_enz = st.selectbox("3' Reverse 限制酶", r_enz_opts, index=0)
                    with p3:
                        an_len = st.slider("退火區長度 (bp)", 18, 27, 21)

                    p_res = design_sticky_end_primers(eng_cdna, f_enz, r_enz, anneal_len=an_len)
                    pm1, pm2, pm3 = st.columns(3)
                    pm1.metric("Forward 引子退火 Tm", f"{p_res['fwd_tm']:.1f} °C")
                    pm2.metric("Reverse 引子退火 Tm", f"{p_res['rev_tm']:.1f} °C")
                    pm3.metric("建議 PCR 退火溫度 (Ta)", f"{p_res['ta_rec']:.1f} °C")

                    st.caption(f"**5' Forward Primer** ({p_res['fwd_len']} nt):")
                    st.code(p_res["fwd_full"], language=None)
                    st.caption(f"**3' Reverse Primer** ({p_res['rev_len']} nt):")
                    st.code(p_res["rev_full"], language=None)

        elif "Gibson" in cloning_method:
            with st.container(border=True):
                st.markdown("**Gibson Assembly / In-Fusion 同源重組引子設計：**")
                g1, g2, g3 = st.columns(3)
                with g1:
                    v5 = st.text_input("載體 5' 端同源重疊序列 (20 nt)", value="CGCCATGGGGATCCGAATTC")
                with g2:
                    v3 = st.text_input("載體 3' 端同源重疊序列 (20 nt)", value="CTCGAGTCTAGAAAGCTTGC")
                with g3:
                    g_an_len = st.slider("退火區長度 (bp)", 18, 27, 21, key="gibson_an_slider")

                g_primers = design_gibson_primers(eng_cdna, vec_5_overlap=v5, vec_3_overlap=v3, anneal_len=g_an_len)
                gm1, gm2, gm3 = st.columns(3)
                gm1.metric("Forward 退火 Tm", f"{g_primers['fwd_tm']:.1f} °C (全長: {g_primers['fwd_tm_full']:.1f} °C)")
                gm2.metric("Reverse 退火 Tm", f"{g_primers['rev_tm']:.1f} °C (全長: {g_primers['rev_tm_full']:.1f} °C)")
                gm3.metric("建議 PCR Ta", f"{g_primers['ta_rec']:.1f} °C")

                st.caption(f"**Forward Gibson Primer** ({g_primers['fwd_len']} nt):")
                st.code(g_primers["fwd_full"], language=None)
                st.caption(f"**Reverse Gibson Primer** ({g_primers['rev_len']} nt):")
                st.code(g_primers["rev_full"], language=None)

        elif "Golden Gate" in cloning_method:
            with st.container(border=True):
                st.markdown("**Type IIS Golden Gate 模組化無痕組裝引子設計：**")
                gg1, gg2, gg3 = st.columns(3)
                with gg1:
                    gg_enz = st.selectbox("Type IIS 酵素", ["BsaI", "BsmBI"], index=0)
                with gg2:
                    oh5 = st.text_input("5' Overhang (4 nt)", value="AATG")
                with gg3:
                    oh3 = st.text_input("3' Overhang (4 nt)", value="GCTT")

                gg_primers = design_golden_gate_primers(eng_cdna, enzyme=gg_enz, overhang_5=oh5, overhang_3=oh3)
                gg_m1, gg_m2, gg_m3 = st.columns(3)
                gg_m1.metric("Forward 退火 Tm", f"{gg_primers['fwd_tm']:.1f} °C")
                gg_m2.metric("Reverse 退火 Tm", f"{gg_primers['rev_tm']:.1f} °C")
                gg_m3.metric("建議 PCR Ta", f"{gg_primers['ta_rec']:.1f} °C")

                st.caption(f"**Golden Gate Forward Primer** ({gg_primers['fwd_len']} nt):")
                st.code(gg_primers["fwd_full"], language=None)
                st.caption(f"**Golden Gate Reverse Primer** ({gg_primers['rev_len']} nt):")
                st.code(gg_primers["rev_full"], language=None)

    # -------------------------------------------------------------
    # Tab 8: 熱力學地景與溫度動力學 (Thermodynamics & Kinetics)
    # -------------------------------------------------------------
    with tab_energy_view:
        model_a_results = calculate_thermodynamics(a_unfolded, a_transition, a_native)
        model_b_results = calculate_thermodynamics(b_unfolded, b_transition, b_native)

        a_x, a_y = generate_energy_curve(a_unfolded, a_transition, a_native)
        b_x, b_y = generate_energy_curve(b_unfolded, b_transition, b_native)

        model_a = {"x": a_x, "y": a_y, "g_unfolded": a_unfolded, "g_transition": a_transition, "g_native": a_native}
        model_b = {"x": b_x, "y": b_y, "g_unfolded": b_unfolded, "g_transition": b_transition, "g_native": b_native}

        sub_2d, sub_3d, sub_kin = st.tabs([t["tab_2d"], t["tab_3d"], t["tab_kinetics"]])

        with sub_2d:
            fig_2d = create_energy_figure(model_a, model_b, lang=lang)
            st.plotly_chart(fig_2d, use_container_width=True, key="energy_2d_chart")

        with sub_3d:
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.3, 1.3, 1.4])
            with ctrl_col1:
                funnel_type = st.selectbox(
                    t["topo_label"], ["smooth", "rugged", "multi"],
                    format_func=lambda x: {"smooth": t["topo_smooth"], "rugged": t["topo_rugged"], "multi": t["topo_multi"]}[x]
                )
            with ctrl_col2:
                target_choice = st.selectbox(
                    t["target_model"], ["Model A", "Model B", "Compare Both"],
                    format_func=lambda x: t["opt_compare_both"] if x == "Compare Both" else x
                )
            with ctrl_col3:
                if funnel_type == "rugged":
                    roughness_amp = st.slider(t["rough_amp"], 0.0, 6.0, 2.5, 0.5)
                    basin_depth = 9.0
                elif funnel_type == "multi":
                    basin_depth = st.slider(t["trap_depth"], 2.0, 20.0, 9.0, 1.0)
                    roughness_amp = 2.5
                else:
                    roughness_amp = 0.0
                    basin_depth = 9.0

            if target_choice == "Compare Both":
                shared_z = [min(a_native, b_native) - 10.0, max(a_unfolded, b_unfolded) + 12.0]
                c_fa, c_fb = st.columns(2)
                with c_fa:
                    fig_3d_a = create_funnel_figure(model_a, funnel_type=funnel_type, roughness_amplitude=roughness_amp, basin_depth=basin_depth, z_range=shared_z, model_label="Model A", height=580, lang=lang)
                    st.plotly_chart(fig_3d_a, use_container_width=True, key="funnel_3d_compare_a")
                with c_fb:
                    fig_3d_b = create_funnel_figure(model_b, funnel_type=funnel_type, roughness_amplitude=roughness_amp, basin_depth=basin_depth, z_range=shared_z, model_label="Model B", height=580, lang=lang)
                    st.plotly_chart(fig_3d_b, use_container_width=True, key="funnel_3d_compare_b")
            else:
                sel_m = model_a if target_choice == "Model A" else model_b
                fig_3d = create_funnel_figure(sel_m, funnel_type=funnel_type, roughness_amplitude=roughness_amp, basin_depth=basin_depth, model_label=target_choice, height=650, lang=lang)
                st.plotly_chart(fig_3d, use_container_width=True, key="funnel_3d_chart")

        with sub_kin:
            k1, k2, k3, k4 = st.columns([1.5, 1.3, 1.1, 1.1])
            with k1:
                t_min, t_max = st.slider(t["temp_range_label"], 250.0, 550.0, (275.0, 420.0), 5.0, key="arrhenius_temp_slider")
            with k2:
                ds_fold = st.slider("ΔS_fold (kJ/mol·K)", -0.50, -0.05, -0.25, 0.05, key="arrhenius_ds_slider")

            sweep_a = simulate_temperature_sweep(model_a_results["delta_g_fold_barrier"], model_a_results["delta_g_unfold_barrier"], delta_s_fold=ds_fold, t_min=t_min, t_max=t_max)
            sweep_b = simulate_temperature_sweep(model_b_results["delta_g_fold_barrier"], model_b_results["delta_g_unfold_barrier"], delta_s_fold=ds_fold, t_min=t_min, t_max=t_max)

            with k3:
                tm_a_str = f"{sweep_a['t_m']:.1f} K ({sweep_a['t_m'] - 273.15:.1f} °C)" if sweep_a["t_m"] else t["tm_none"]
                st.metric(t["tm_metric_a"], tm_a_str)
            with k4:
                tm_b_str = f"{sweep_b['t_m']:.1f} K ({sweep_b['t_m'] - 273.15:.1f} °C)" if sweep_b["t_m"] else t["tm_none"]
                st.metric(t["tm_metric_b"], tm_b_str)

            fig_kinetics = create_kinetics_sweep_figure(sweep_a, sweep_b, lang=lang)
            st.plotly_chart(fig_kinetics, use_container_width=True, key="kinetics_sweep_chart")

        # 熱力學結論面板 (無重複宣告)
        with st.container(border=True):
            st.markdown(f"**{t['metrics_header']}**")
            dg_a = model_a_results["delta_g_fold"]
            dg_b = model_b_results["delta_g_fold"]
            ea_a = model_a_results["delta_g_fold_barrier"]
            ea_b = model_b_results["delta_g_fold_barrier"]

            res_col_a, res_col_b = st.columns(2)
            with res_col_a:
                st.markdown("##### 🔵 Model A (基準)")
                c1, c2 = st.columns(2)
                c1.metric(t["dg_fold"], f"{dg_a:.1f} kJ/mol", delta=f"{dg_a - dg_b:+.1f} kJ/mol", delta_color="inverse")
                c2.metric(t["ea_fold"], f"{ea_a:.1f} kJ/mol", delta=f"{ea_a - ea_b:+.1f} kJ/mol", delta_color="inverse")
            with res_col_b:
                st.markdown("##### 🟠 Model B (突變/比較)")
                c3, c4 = st.columns(2)
                c3.metric(t["dg_fold"], f"{dg_b:.1f} kJ/mol", delta=f"{dg_b - dg_a:+.1f} kJ/mol", delta_color="inverse")
                c4.metric(t["ea_fold"], f"{ea_b:.1f} kJ/mol", delta=f"{ea_b - ea_a:+.1f} kJ/mol", delta_color="inverse")

    # -------------------------------------------------------------
    # Tab 9: 成果匯出與 GenBank 下載 (Export Report & GenBank)
    # -------------------------------------------------------------
    with tab_export_view:
        st.markdown("### 📥 研究報告與質體圖譜成果匯出")
        st.caption("將完整的結構生物物理指標、熱力學能量地景、突變預測、引子與序列匯出為獨立保存的網頁或 SnapGene 相容格式。")

        # 準備資料包
        export_bundle = {
            "title": struct_title,
            "sequence": target_bp_chain["sequence"],
            "bp_stats": bp_stats,
            "bio_stats": analyze_fasta_biochem(target_bp_chain["sequence"]),
            "nt_stats": {
                "cdna_len": nt_len,
                "gc_pct": gc_total,
                "gc3_pct": gc3_pct,
                "cai": cai_val,
                "dg_5prime": dg_5prime,
                "cdna_seq": cdna_seq
            },
            "primers": {
                "fwd_full": p_res["fwd_full"] if "p_res" in locals() else "N/A",
                "rev_full": p_res["rev_full"] if "p_res" in locals() else "N/A",
                "fwd_tm": p_res["fwd_tm"] if "p_res" in locals() else 0.0,
                "rev_tm": p_res["rev_tm"] if "p_res" in locals() else 0.0,
                "fwd_len": p_res["fwd_len"] if "p_res" in locals() else 0,
                "rev_len": p_res["rev_len"] if "p_res" in locals() else 0
            },
            "thermo_stats": {
                "tm_str": tm_a_str if "tm_a_str" in locals() else "N/A"
            },
            "ss_comp": {
                "h_pct": h_pct if "h_pct" in locals() else 0.0,
                "e_pct": e_pct if "e_pct" in locals() else 0.0,
                "c_pct": c_pct if "c_pct" in locals() else 0.0,
                "fold_class": fold_class if "fold_class" in locals() else "Unknown"
            },
            "ss_candidates": ss_analysis.get("candidates", []) if "ss_analysis" in locals() else []
        }

        report_html_content = generate_standalone_report_html(export_bundle)

        genbank_text = generate_genbank_record(
            struct_title.split()[0].replace(".", "_"),
            cdna_seq,
            target_bp_chain["sequence"],
            features=eng_construct.get("features", []) if "eng_construct" in locals() else []
        )

        d1, d2 = st.columns(2)
        with d1:
            with st.container(border=True):
                st.markdown("#### 📄 1. 獨立可儲存網頁報告 (HTML / PDF)")
                st.caption("內建響應式現代 CSS 樣式，可離線雙擊開啟，並支援直接按 Ctrl+P 列印成美觀 PDF 報告。")
                st.download_button(
                    "💾 下載獨立網頁報告 (.html)",
                    data=report_html_content.encode("utf-8"),
                    file_name=f"{struct_title.split()[0]}_Biophysics_Report.html",
                    mime="text/html",
                    use_container_width=True
                )

        with d2:
            with st.container(border=True):
                st.markdown("#### 🧬 2. GenBank 標準質體圖譜 (.gb)")
                st.caption("包含 CDS 區域、標籤與切位特徵註解，可直接拉入 SnapGene、Benchling 或 Geneious。")
                st.download_button(
                    "💾 下載 GenBank 質體檔 (.gb)",
                    data=genbank_text.encode("utf-8"),
                    file_name=f"{struct_title.split()[0]}_construct.gb",
                    mime="text/plain",
                    use_container_width=True
                )

else:
    st.info("請於左側側邊欄選擇輸入結構（上傳 PDB、輸入 RCSB 代碼或載入內建範例）。")

st.caption(t["footer_note"])
