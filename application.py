def apply_biophys_to_model_a(est_u, est_ts, est_n):
    """在側邊欄元件建立前，透過 on_click 回呼安全同步 Model A 的所有狀態變數"""
    st.session_state["a_u"] = est_u
    st.session_state["a_ts"] = est_ts
    st.session_state["a_n"] = est_n

    st.session_state["slider_a_u"] = est_u
    st.session_state["slider_a_ts"] = est_ts
    st.session_state["slider_a_n"] = est_n

    st.session_state["num_a_u"] = est_u
    st.session_state["num_a_ts"] = est_ts
    st.session_state["num_a_n"] = est_n

import numpy as np

import streamlit as st

from data_loader import load_pdb
from data_loader import load_pdb, load_sample_ubiquitin

from thermodynamics import (
    calculate_thermodynamics,
    generate_energy_curve,
    simulate_temperature_sweep,
    analyze_structural_biophysics
)

from visualization import (
    create_energy_figure,
    create_funnel_figure,
    create_ca_backbone_figure,
    create_contact_map_figure,
    create_ramachandran_figure,
    create_kinetics_sweep_figure,
    create_nucleotide_analysis_figure
)

# 1. 頁面設定 (必須為第一個 st 指令)
st.set_page_config(
    page_title="Protein Thermodynamics Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* 1. 讓 st.metric 指標標題與數值自動換行，禁止變成 ... 吃字 */
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        font-size: clamp(0.78rem, 1.05vw, 0.95rem) !important;
    }
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow-wrap: break-word !important;
        font-size: clamp(1.15rem, 1.8vw, 1.65rem) !important;
        line-height: 1.2 !important;
    }

    /* 2. 讓按鈕與分頁標籤在窄視窗下自動換行不裁切 */
    .stButton > button p {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.3 !important;
    }
    button[data-baseweb="tab"] p {
        white-space: normal !important;
        font-size: clamp(0.8rem, 1vw, 0.95rem) !important;
    }

    /* 3. 主容器左右內距自適應，避免小螢幕內容被擠壓 */
    .block-container {
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
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
        "tab_nt": "基因與密碼子優化",
        "tab_cloning": "分子選殖與引子設計",
        "color_by_codon_gc": "依推演基因密碼子 GC% 映射 (結構—基因聯動)",
        "nt_plot_title": "推演編碼區 cDNA 局部 GC% 與鹼基組成分析 (視窗 = {win} nt)",
        "nt_sub_left": "滑動視窗局部 GC 含量分布 (5' → 3')",
        "nt_sub_right": "四種核苷酸鹼基組成統計",
        "nt_xaxis_pos": "核苷酸序列位置 (nt)",
        "nt_yaxis_gc": "局部 GC 含量 (%)",
        "nt_yaxis_cnt": "鹼基數量 (nt)",
        "cm_mode_label": "顯示模式",
        "cm_mode_binary": "二值化接觸圖 (Contact Map)",
        "cm_mode_dist": "連續距離熱圖 (Distance Matrix)",
        "cm_cutoff_label": "空間接觸閾值 (Cutoff, Å)",
        "color_by_ribbon": "🎀 二級結構飄帶模式 (Helix / Sheet Ribbon)",
        "ss_summary_title": "📊 二級結構組成與 SCOP 拓撲分類分析",
        "ss_helix_pct": "α-螺旋比例 (α-Helix)",
        "ss_sheet_pct": "β-折板比例 (β-Sheet)",
        "ss_coil_pct": "無規捲曲 / 環區 (Coil)",
        "ss_fold_class": "結構域拓撲類別",
        "class_all_alpha": "全 α-螺旋型 (All-α)",
        "class_all_beta": "全 β-折板型 (All-β)",
        "class_alpha_beta": "α/β 混合型 (α + β / α/β)",
        "class_coil": "無規捲曲主導型 (Disordered / Coil)",
        "biophys_header": "🔬 結構拓撲與生物物理指標（驅動熱力學模型）",
        "biophys_chain_select": "選擇分析胜肽鏈",
        "metric_rco": "相對接觸序 (RCO)",
        "metric_rco_help": "衡量三級結構的拓撲複雜度。RCO 越高，長程接觸越多，折疊活化能障（ΔG‡）越高。",
        "metric_lr": "長程接觸比例 (|i-j| ≥ 12)",
        "metric_rg": "迴轉半徑 (Rg)",
        "metric_core": "疏水核心埋藏率",
        "metric_core_help": "非極性殘基（A,V,I,L,M,F,Y,W）位於結構核心（< 0.75 Rg）的比例，反映疏水塌陷緊密度。",
        "btn_sync_model_a": "⚡ 將此結構推算參數套用至 Model A（TS = {ts:.1f}, N = {n:.1f} kJ/mol）",
        "sync_success_toast": "已將 {chain_label} 的物理推算參數同步至 Model A！",
        "btn_load_sample": "🧪 一鍵載入範例：Ubiquitin (1UBQ)",
        "btn_clear_sample": "清除範例結構",
        "sample_active_badge": "目前檢視內建模型蛋白質：**Human Ubiquitin (PDB: 1UBQ, 76 aa)**",
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
        "tab_nt": "Gene & Codon Opt",
        "tab_cloning": "Cloning & Primers",
        "color_by_codon_gc": "By Deduced Codon GC% (Structure-Gene Link)",
        "nt_plot_title": "Deduced Coding cDNA Local GC% & Base Composition (Window = {win} nt)",
        "nt_sub_left": "Sliding-Window Local GC% Profile (5' → 3')",
        "nt_sub_right": "Nucleotide Base Composition",
        "nt_xaxis_pos": "Nucleotide Position (nt)",
        "nt_yaxis_gc": "Local GC Content (%)",
        "nt_yaxis_cnt": "Base Count (nt)",
        "cm_mode_label": "Display Mode",
        "cm_mode_binary": "Binary Contact Map",
        "cm_mode_dist": "Continuous Distance Matrix",
        "cm_cutoff_label": "Contact Distance Cutoff (Å)",
        "color_by_ribbon": "🎀 Secondary Structure Ribbon (Helix / Sheet)",
        "ss_summary_title": "📊 Secondary Structure Content & Fold Classification",
        "ss_helix_pct": "α-Helix Content",
        "ss_sheet_pct": "β-Sheet Content",
        "ss_coil_pct": "Coil / Loop Content",
        "ss_fold_class": "SCOP Fold Class",
        "class_all_alpha": "All-α Protein",
        "class_all_beta": "All-β Protein",
        "class_alpha_beta": "α/β Mixed Fold",
        "class_coil": "Coil / Loop Dominated",
        "biophys_header": "🔬 Structural Biophysics & Topology Metrics (Model Coupling)",
        "biophys_chain_select": "Target Chain for Biophysics",
        "metric_rco": "Relative Contact Order (RCO)",
        "metric_rco_help": "Topological complexity metric. Higher RCO indicates more non-local contacts and a higher folding barrier (ΔG‡).",
        "metric_lr": "Long-Range Contacts (|i-j| ≥ 12)",
        "metric_rg": "Radius of Gyration (Rg)",
        "metric_core": "Hydrophobic Core Burial",
        "metric_core_help": "Fraction of nonpolar residues (A,V,I,L,M,F,Y,W) buried within 0.75 Rg of the center of mass.",
        "btn_sync_model_a": "⚡ Apply Estimated Parameters to Model A (TS = {ts:.1f}, N = {n:.1f} kJ/mol)",
        "sync_success_toast": "Synced biophysical parameters from {chain_label} to Model A!",
        "btn_load_sample": "🧪 Load Sample: Ubiquitin (1UBQ)",
        "btn_clear_sample": "Clear Sample Structure",
        "sample_active_badge": "Active Built-in Model Protein: **Human Ubiquitin (PDB: 1UBQ, 76 aa)**",
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

RESTRICTION_SITES = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "NdeI": "CATATG",
    "XhoI": "CTCGAG",
    "NotI": "GCGGCCGC",
    "NcoI": "CCATGG",
    "SalI": "GTCGAC"
}


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

# 宿主相對同義密碼子適應度權重 (RSCU / CAI 參考權重)
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


def compute_cai_and_5prime_dg(mrna_seq, strategy="ecoli"):
    host = "human" if strategy == "human" else "ecoli"
    weights_tbl = CODON_WEIGHTS[host]
    codons = [mrna_seq[i:i+3] for i in range(0, len(mrna_seq) - 2, 3)]
    valid_w = [weights_tbl[c] for c in codons if c in weights_tbl]
    cai = float(np.exp(np.mean(np.log(valid_w)))) if valid_w else 0.0

    # 評估 5' 端前 30 nt 局部折疊自由能估算 (kcal/mol, 負值越大代表髮夾越強、越阻礙轉譯起始)
    head_30 = mrna_seq[:30]
    gc_head = head_30.count("G") + head_30.count("C")
    au_head = head_30.count("A") + head_30.count("U")
    est_5prime_dg = round(-0.55 * gc_head - 0.15 * au_head + 4.2, 2)
    return cai, est_5prime_dg


def reverse_complement(dna_seq):
    comp = str.maketrans("ACGTURYKMSWBDHVN", "TGCAAYRMKSWVHDBN")
    return dna_seq.upper().translate(comp)[::-1]


def calc_primer_tm(seq):
    s = seq.upper()
    n = len(s)
    if n == 0:
        return 0.0
    gc = s.count("G") + s.count("C")
    if n < 14:
        return round(2.0 * (n - gc) + 4.0 * gc, 1)
    return round(64.9 + 41.0 * (gc - 16.4) / n, 1)


def design_cloning_primers(cdna_seq, fwd_enz, rev_enz, anneal_len=21):
    # 保護鹼基 + 限制酶辨識序列 + 退火區段
    clamp_5 = "GGAATT"
    fwd_site = RESTRICTION_SITES[fwd_enz]
    rev_site = RESTRICTION_SITES[rev_enz]

    fwd_anneal = cdna_seq[:anneal_len]
    fwd_full = f"{clamp_5}{fwd_site}{fwd_anneal}"

    # 反向引子扣除最後 3 bp 終止密碼子（利於接 C-terminal His-tag）或直接從末端設計
    core_3prime = cdna_seq[:-3] if cdna_seq.endswith(("TAA", "TAG", "TGA")) else cdna_seq
    rev_anneal = reverse_complement(core_3prime[-anneal_len:])
    rev_full = f"{clamp_5}{rev_site}{rev_anneal}"

    fwd_tm = calc_primer_tm(fwd_anneal)
    rev_tm = calc_primer_tm(rev_anneal)
    ta_rec = round(min(fwd_tm, rev_tm) - 3.0, 1)

    return {
        "fwd_full": fwd_full,
        "fwd_anneal": fwd_anneal,
        "fwd_tm": fwd_tm,
        "rev_full": rev_full,
        "rev_anneal": rev_anneal,
        "rev_tm": rev_tm,
        "ta_rec": ta_rec
    }

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

    # 二分法求解理論等電點 (pI)
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

    # pH 7.4 淨電荷估算
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

# =====================================
# 側邊欄：語言切換、PDB 上傳與雙模式參數調整
# =====================================
with st.sidebar:
    lang_choice = st.radio(
        "語言 / Language",
        ["繁體中文(臺灣)", "English"],
        horizontal=True
    )
    lang = "zh" if lang_choice == "繁體中文(臺灣)" else "en"
    t = TEXTS[lang]

    st.divider()
    st.subheader(t["sidebar_struct"])
    uploaded_pdb = st.file_uploader(
        t["upload_label"],
        type=["pdb"],
        help=t["upload_help"]
    )
    if "use_sample_1ubq" not in st.session_state:
        st.session_state["use_sample_1ubq"] = False

    if uploaded_pdb is not None:
        st.session_state["use_sample_1ubq"] = False

    if not st.session_state["use_sample_1ubq"]:
        if st.button(t["btn_load_sample"], use_container_width=True):
            st.session_state["use_sample_1ubq"] = True
            st.rerun()
    else:
        if st.button(t["btn_clear_sample"], use_container_width=True):
            st.session_state["use_sample_1ubq"] = False
            st.rerun()
    
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
if uploaded_pdb is not None or st.session_state.get("use_sample_1ubq", False):
    try:
        if uploaded_pdb is not None:
            results = load_pdb(uploaded_pdb)
            struct_title = uploaded_pdb.name
        else:
            results = load_sample_ubiquitin()
            struct_title = "1UBQ (Human Ubiquitin)"
            st.info(t["sample_active_badge"])

        total_chains = len(results)
        total_residues = sum(item["residue_count"] for item in results)
        est_delta_g = -0.25 * total_residues

        st.subheader(f"{t['struct_summary']}: `{struct_title}`")

        # =====================================
        # 結構生物物理指標計算與一鍵套用至 Model A
        # =====================================
        with st.container(border=True):
            bp_top1, bp_top2 = st.columns([2.2, 1.3])
            with bp_top1:
                st.markdown(f"**{t['biophys_header']}**")
            with bp_top2:
                bp_chain_idx = st.selectbox(
                    t["biophys_chain_select"],
                    list(range(len(results))),
                    format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                    label_visibility="collapsed",
                    key="biophys_chain_selector"
                )

            target_bp_chain = results[bp_chain_idx]
            bp_stats = analyze_structural_biophysics(
                target_bp_chain["ca_coords"],
                target_bp_chain["sequence"]
            )

            if bp_stats is not None:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric(
                    t["metric_rco"],
                    f"{bp_stats['rco']:.3f}",
                    help=t["metric_rco_help"]
                )
                b2.metric(
                    t["metric_lr"],
                    f"{bp_stats['long_range_ratio'] * 100:.1f}%"
                )
                b3.metric(
                    t["metric_rg"],
                    f"{bp_stats['rg']:.2f} Å"
                )
                b4.metric(
                    t["metric_core"],
                    f"{bp_stats['core_burial_ratio'] * 100:.1f}%",
                    help=t["metric_core_help"]
                )

                btn_label = t["btn_sync_model_a"].format(
                    ts=bp_stats["est_ts"],
                    n=bp_stats["est_n"]
                )
                btn_label = t["btn_sync_model_a"].format(
                    ts=bp_stats["est_ts"],
                    n=bp_stats["est_n"]
                )
                st.button(
                    btn_label,
                    use_container_width=True,
                    key="btn_apply_biophys_a",
                    on_click=apply_biophys_to_model_a,
                    args=(bp_stats["est_u"], bp_stats["est_ts"], bp_stats["est_n"])
                )
        
        # 雙分頁：3D Cα 骨架模型 vs FASTA 序列
        subtab_3d, subtab_contact, subtab_rama, subtab_seq, subtab_nt, subtab_cloning = st.tabs([
            t["tab_ca_3d"],
            t["tab_contact"],
            t["tab_rama"],
            t["tab_fasta"],
            t["tab_nt"],
            t["tab_cloning"]
        ])

        with subtab_3d:
            available_models = sorted(list(set(item["model"] for item in results)))
            ca_col1, ca_col2, ca_col3 = st.columns([1.0, 1.6, 1.0])

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
                    ["ribbon", "chain", "sequence", "codon_gc"],
                    format_func=lambda x: {
                        "ribbon": t["color_by_ribbon"],
                        "chain": t["color_by_chain"],
                        "sequence": t["color_by_seq"],
                        "codon_gc": t["color_by_codon_gc"]
                    }[x],
                    key="ca_color_select"
                )

            with ca_col3:
                st.write("")
                st.write("")
                show_nodes = st.checkbox(
                    t["show_ca_nodes"],
                    value=(ca_color != "ribbon"),
                    key="ca_nodes_chk"
                )

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

            model_chains = [item for item in results if item["model"] == chosen_pdb_model]
            all_ss = [s for item in model_chains for s in item.get("ss_list", [])]
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

        with subtab_contact:
            cm_col1, cm_col2, cm_col3 = st.columns([1.4, 1.3, 1.3])
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
                    t["cm_mode_label"],
                    ["binary", "distance"],
                    format_func=lambda x: t["cm_mode_binary"] if x == "binary" else t["cm_mode_dist"],
                    key="cm_mode_select"
                )

            with cm_col3:
                cm_cutoff = st.slider(
                    t["cm_cutoff_label"],
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
                map_mode=cm_mode,
                lang=lang
            )
            st.plotly_chart(fig_cm, use_container_width=True, key="pdb_contact_map_chart")

            cm_coords = np.array(results[chosen_chain_idx]["ca_coords"], dtype=float)
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
                    cm2.metric("局部接觸 (3 ≤ |i-j| < 12)" if lang == "zh" else "Local Contacts", f"{n_local} ({100.0 - long_pct:.1f}%)")
                    cm3.metric("長程三級接觸 (|i-j| ≥ 12)" if lang == "zh" else "Long-Range Contacts", f"{n_long} ({long_pct:.1f}%)")

                    if lang == "zh":
                        if long_pct >= 45.0:
                            st.info(f"📋 **拓撲診斷：** 長程跨結構域接觸佔比達 **{long_pct:.1f}%**（閾值 {cm_cutoff:.1f} Å），顯示該胜肽鏈具備緊密球狀核心與高協同折疊（Cooperative Folding）特徵。")
                        else:
                            st.info(f"📋 **拓撲診斷：** 接觸訊號集中於對角線附近（局部接觸佔 **{100.0 - long_pct:.1f}%**），反映結構以連續二級結構元件或伸展螺旋束為主。")
                    else:
                        if long_pct >= 45.0:
                            st.info(f"📋 **Topology Diagnosis:** High long-range tertiary contact share (**{long_pct:.1f}%** at {cm_cutoff:.1f} Å cutoff), indicating a compact globular core and cooperative folding.")
                        else:
                            st.info(f"📋 **Topology Diagnosis:** Contacts cluster near the diagonal (**{100.0 - long_pct:.1f}%** local), typical of extended or helical-bundle topologies.")

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

            with rama_col2:
                if total_angles > 0:
                    alpha_cnt = sum(1 for d in r_list if -110 <= d["phi"] <= -35 and -70 <= d["psi"] <= -15)
                    beta_cnt = sum(1 for d in r_list if -165 <= d["phi"] <= -45 and 90 <= d["psi"] <= 175)
                    other_cnt = total_angles - alpha_cnt - beta_cnt

                    s1, s2, s3 = st.columns(3)
                    s1.metric(t["rama_alpha"], f"{alpha_cnt / total_angles * 100:.1f}%")
                    s2.metric(t["rama_beta"], f"{beta_cnt / total_angles * 100:.1f}%")
                    s3.metric(t["rama_other"], f"{other_cnt / total_angles * 100:.1f}%")

            fig_rama = create_ramachandran_figure(selected_rama_item, lang=lang)
            st.plotly_chart(fig_rama, use_container_width=True, key="pdb_rama_chart")

            if total_angles > 0:
                outliers = []
                favored_cnt = 0
                for idx_r, d in enumerate(r_list):
                    phi = d.get("phi", 0.0)
                    psi = d.get("psi", 0.0)
                    res_name = str(d.get("resn") or d.get("resname") or d.get("aa") or d.get("label") or f"Res{idx_r + 1}").strip().upper()
                    res_id = d.get("resi") or d.get("resseq") or d.get("idx") or (idx_r + 1)
                    tag_str = res_name if any(ch.isdigit() for ch in res_name) else f"{res_name}-{res_id}"

                    in_alpha = (-140 <= phi <= -30) and (-80 <= psi <= 10)
                    in_beta = (-180 <= phi <= -40) and (80 <= psi <= 180)
                    in_left = (30 <= phi <= 100) and (0 <= psi <= 90)
                    is_gly = res_name.startswith("GLY") or res_name.startswith("G-") or res_name == "G"

                    if in_alpha or in_beta or in_left or is_gly:
                        favored_cnt += 1
                    else:
                        outliers.append(f"{tag_str} (φ={phi:.1f}°, ψ={psi:.1f}°)")

                fav_pct = favored_cnt / total_angles * 100.0
                with st.container(border=True):
                    if lang == "zh":
                        st.markdown(f"**🔬 主鏈立體化學品質診斷（核心允許區佔比：{fav_pct:.1f}%）**")
                        if not outliers:
                            st.success("所有非甘胺酸（Non-Glycine）殘基皆落入立體化學允許構象區，無主鏈空間位阻異常。")
                        else:
                            st.warning(f"偵測到 **{len(outliers)}** 個非甘胺酸構象張力/異常殘基（常見於緊密轉角或活性中心）：`{', '.join(outliers[:8])}`" + (" ..." if len(outliers) > 8 else ""))
                    else:
                        st.markdown(f"**🔬 Backbone Stereochemical Quality (Favored / Gly-Allowed: {fav_pct:.1f}%)**")
                        if not outliers:
                            st.success("All non-glycine residues fall within sterically allowed Ramachandran regions.")
                        else:
                            st.warning(f"Detected **{len(outliers)}** non-glycine conformational outliers (often tight turns or strained active sites): `{', '.join(outliers[:8])}`" + (" ..." if len(outliers) > 8 else ""))

        with subtab_seq:
            for item in results:
                seq_str = item["sequence"]
                header_tag = f">Model_{item['model']}_Chain_{item['chain']} | {item['residue_count']} aa"
                fasta_block = f"{header_tag}\n{seq_str}"
                bio_stats = analyze_fasta_biochem(seq_str)

                with st.container(border=True):
                    st.markdown(
                        f"**Model {item['model']} | Chain {item['chain']}** "
                        f"(`{item['residue_count']}` residues)"
                    )
                    if bio_stats is not None:
                        f1, f2, f3, f4 = st.columns(4)
                        f1.metric("分子量 (MW)" if lang == "zh" else "Mol. Weight", f"{bio_stats['mw_kda']:.2f} kDa")
                        f2.metric("理論等電點 (pI)" if lang == "zh" else "Theoretical pI", f"{bio_stats['pi']:.2f}")
                        f3.metric("GRAVY 疏水性" if lang == "zh" else "GRAVY Index", f"{bio_stats['gravy']:+.3f}")
                        f4.metric("pH 7.4 淨電荷" if lang == "zh" else "Net Charge (pH 7.4)", f"{bio_stats['net_charge']:+.1f}")

                        if lang == "zh":
                            sol_msg = "整體親水、重組表現可溶性佳" if bio_stats["gravy"] < 0 else "整體偏疏水（可能為膜蛋白或易形成包涵體）"
                            buf_msg = "帶負電（建議使用陰離子交換層析或 pH < pI 緩衝液）" if bio_stats["net_charge"] < -1 else ("帶正電（建議使用陽離子交換層析）" if bio_stats["net_charge"] > 1 else "接近電中性（需注意等電點沉澱風險）")
                            st.caption(f"🧪 **生化表現診斷：** {sol_msg}；生理 pH 下{buf_msg}；芳香族殘基佔比 {bio_stats['arom_pct']:.1f}%。")
                        else:
                            sol_msg = "Hydrophilic profile favoring soluble expression" if bio_stats["gravy"] < 0 else "Hydrophobic profile (membrane-associated or inclusion-body prone)"
                            st.caption(f"🧪 **Biochemical Diagnosis:** {sol_msg}; aromatic content {bio_stats['arom_pct']:.1f}% for A280 quantification.")

                    st.code(fasta_block, language=None)

        with subtab_nt:
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
                    t["nt_strategy"],
                    ["ecoli", "human", "deg"],
                    format_func=lambda x: {
                        "ecoli": "E. coli Codon-Optimized",
                        "human": "Homo sapiens Codon-Optimized",
                        "deg": t["nt_deg"]
                    }[x],
                    key="nt_strat_select"
                )
            with nt_col3:
                st.write("")
                st.write("")
                add_ends = st.checkbox(t["nt_include_stops"], value=True, key="nt_ends_chk")

            target_nt_item = results[nt_chain_idx]
            mrna_seq, cdna_seq = reverse_translate_sequence(
                target_nt_item["sequence"],
                strategy=nt_strat,
                add_start_stop=add_ends
            )

            nt_len = len(cdna_seq)
            gc_total = (cdna_seq.count("G") + cdna_seq.count("C")) / nt_len * 100.0 if nt_len > 0 else 0.0
            wobble_bases = cdna_seq[2::3]
            gc3_pct = (wobble_bases.count("G") + wobble_bases.count("C")) / len(wobble_bases) * 100.0 if len(wobble_bases) > 0 else 0.0
            mw_kda = (nt_len * 617.96 + 36.04) / 1000.0

            nm1, nm2, nm3, nm4 = st.columns(4)
            nm1.metric("cDNA Length", f"{nt_len} bp")
            nm2.metric("Overall GC%", f"{gc_total:.1f}%")
            nm3.metric("Wobble GC3%", f"{gc3_pct:.1f}%")
            nm4.metric("dsDNA MW", f"{mw_kda:.2f} kDa")

            cai_val, dg_5prime = compute_cai_and_5prime_dg(mrna_seq, strategy=nt_strat)
            em1, em2 = st.columns(2)
            em1.metric("密碼子適應指數 (CAI)" if lang == "zh" else "Codon Adaptation Index (CAI)", f"{cai_val:.3f}")
            em2.metric("mRNA 5' 端前 30nt 折疊 ΔG" if lang == "zh" else "5' mRNA Folding ΔG (30 nt)", f"{dg_5prime:.2f} kcal/mol")

            if lang == "zh":
                if dg_5prime < -8.0:
                    st.warning(f"⚠️ **轉譯起始風險：** 5' 端前 30 nt 局部自由能為 **{dg_5prime:.2f} kcal/mol**（GC 偏高），可能形成穩定髮夾結構阻礙核糖體結合。")
                else:
                    st.success(f"✅ **轉譯起始順暢：** 5' 端局部自由能為 **{dg_5prime:.2f} kcal/mol**（結構鬆散），利於核糖體 30S 次單元結合與起始轉譯。")
            else:
                if dg_5prime < -8.0:
                    st.warning(f"⚠️ **Translation Initiation Risk:** 5' mRNA ΔG is **{dg_5prime:.2f} kcal/mol**, which may form secondary hairpins blocking ribosome entry.")
                else:
                    st.success(f"✅ **Favorable Initiation:** 5' mRNA ΔG is **{dg_5prime:.2f} kcal/mol**, minimizing secondary structure hindrance.")

            fig_nt = create_nucleotide_analysis_figure(cdna_seq, window=15, lang=lang)
            st.plotly_chart(fig_nt, use_container_width=True, key="pdb_nt_analysis_chart")

            st.caption(t["nt_mrna_label"])
            st.code(mrna_seq, language=None)
            st.caption(t["nt_cdna_label"])
            st.code(cdna_seq, language=None)

        with subtab_cloning:
            cl_col1, cl_col2 = st.columns([1.4, 1.4])
            with cl_col1:
                cl_chain_idx = st.selectbox(
                    t["select_chain"],
                    list(range(len(results))),
                    format_func=lambda idx: f"Model {results[idx]['model']} - Chain {results[idx]['chain']} ({results[idx]['residue_count']} aa)",
                    key="cl_chain_select"
                )
            with cl_col2:
                cl_host = st.selectbox(
                    t["nt_strategy"],
                    ["ecoli", "human"],
                    format_func=lambda x: "E. coli Expression Construct" if x == "ecoli" else "Homo sapiens Expression Construct",
                    key="cl_host_select"
                )

            target_cl_item = results[cl_chain_idx]
            _, cl_cdna = reverse_translate_sequence(
                target_cl_item["sequence"],
                strategy=cl_host,
                add_start_stop=True
            )
            hits, zero_cutters = scan_restriction_sites(cl_cdna)

            with st.container(border=True):
                if lang == "zh":
                    st.markdown("**✂️ 1. 常見質體多選殖位點 (MCS) 限制酶切位掃描**")
                    if hits:
                        hit_str = " ｜ ".join([f"**{enz}** (位置: {', '.join(map(str, pos))} bp)" for enz, pos in hits.items()])
                        st.warning(f"偵測到內部限制酶切位（選殖時請避開）：{hit_str}")
                    else:
                        st.success("序列內部未發現常見 MCS 限制酶切位，所有常用酵素皆可安全使用。")
                    st.caption(f"0-cutter 安全可用酵素：{', '.join(zero_cutters)}")
                else:
                    st.markdown("**✂️ 1. Plasmid MCS Restriction Enzyme Site Scan**")
                    if hits:
                        hit_str = " | ".join([f"**{enz}** (pos: {', '.join(map(str, pos))} bp)" for enz, pos in hits.items()])
                        st.warning(f"Internal restriction sites detected (avoid for cloning): {hit_str}")
                    else:
                        st.success("No internal MCS restriction sites detected.")
                    st.caption(f"Safe 0-cutter enzymes: {', '.join(zero_cutters)}")

            if len(zero_cutters) >= 2:
                with st.container(border=True):
                    st.markdown("**🧬 2. PCR 黏性末端選殖引子自動設計器 (Sticky-End Primer Designer)**" if lang == "zh" else "**🧬 2. Automated Sticky-End PCR Primer Designer**")
                    p_col1, p_col2, p_col3 = st.columns([1.2, 1.2, 1.4])
                    with p_col1:
                        fwd_enz = st.selectbox("5' Forward 限制酶" if lang == "zh" else "5' Forward Enzyme", zero_cutters, index=0, key="primer_fwd_enz")
                    with p_col2:
                        rev_options = [e for e in zero_cutters if e != fwd_enz]
                        rev_enz = st.selectbox("3' Reverse 限制酶" if lang == "zh" else "3' Reverse Enzyme", rev_options, index=0, key="primer_rev_enz")
                    with p_col3:
                        anneal_len = st.slider("模板退火區長度 (bp)" if lang == "zh" else "Annealing Length (bp)", 18, 27, 21, key="primer_anneal_slider")

                    p_res = design_cloning_primers(cl_cdna, fwd_enz, rev_enz, anneal_len=anneal_len)

                    pm1, pm2, pm3 = st.columns(3)
                    pm1.metric("Forward Primer 退火 Tm", f"{p_res['fwd_tm']:.1f} °C")
                    pm2.metric("Reverse Primer 退火 Tm", f"{p_res['rev_tm']:.1f} °C")
                    pm3.metric("建議 PCR 退火溫度 (Ta)" if lang == "zh" else "Recommended PCR Ta", f"{p_res['ta_rec']:.1f} °C")

                    st.caption(f"**5' Forward Primer** (`5' → 3'`, {len(p_res['fwd_full'])} nt | 保護鹼基 + `{fwd_enz}` + 起始區):")
                    st.code(p_res["fwd_full"], language=None)

                    st.caption(f"**3' Reverse Primer** (`5' → 3'`, {len(p_res['rev_full'])} nt | 保護鹼基 + `{rev_enz}` + 反向互補區，已去終止密碼子便於接 His-Tag):")
                    st.code(p_res["rev_full"], language=None)
                    
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
    fig_2d = create_energy_figure(model_a, model_b, lang=lang)
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
                height=600,
                lang=lang
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
                height=600,
                lang=lang
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
            height=720,
            lang=lang
        )
        st.plotly_chart(fig_3d, use_container_width=True, key="funnel_3d_chart")

with tab_kinetics:
    k_col1, k_col2, k_col3, k_col4 = st.columns([1.5, 1.3, 1.1, 1.1])

    with k_col1:
        t_min, t_max = st.slider(
            t["temp_range_label"],
            min_value=250.0,
            max_value=550.0,
            value=(275.0, 420.0),
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

    fig_kinetics = create_kinetics_sweep_figure(sweep_a, sweep_b, lang=lang)
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