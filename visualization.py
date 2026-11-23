import numpy as np
from datetime import datetime

import plotly.graph_objects as go

from plotly.subplots import make_subplots


# =====================================
# 圖表多語系字典 (繁體中文 / English)
# =====================================
PLOT_TEXTS = {
    "zh": {
        "ca_gene_gc_colorbar": "密碼子<br>GC%",
        "ca_gene_gc_title": "Model {model} 3D 結構—基因密碼子 GC% 聯動圖譜",
        "nt_plot_title": "推演編碼區 cDNA 局部 GC% 與鹼基組成分析 (視窗 = {win} nt)",
        "nt_sub_left": "滑動視窗局部 GC 含量分布 (5' → 3')",
        "nt_sub_right": "四種核苷酸鹼基組成統計",
        "nt_xaxis_pos": "核苷酸序列位置 (nt)",
        "nt_yaxis_gc": "局部 GC 含量 (%)",
        "nt_yaxis_cnt": "鹼基數量 (nt)",
        "energy_title": "蛋白質折疊自由能剖面比較",
        "energy_xaxis": "反應座標 (Reaction Coordinate)",
        "energy_yaxis": "吉布斯自由能 (kJ/mol)",
        "energy_legend": "模型",
        "states_suffix": "狀態節點",
        "funnel_smooth": "平滑折疊漏斗 (Smooth Funnel)",
        "funnel_rugged": "粗糙能量地景 (Rugged Landscape)",
        "funnel_multi": "多盆地錯誤折疊地景 (Multi-basin)",
        "funnel_x": "構形座標 1",
        "funnel_y": "構形座標 2",
        "funnel_z": "有效自由能 (kJ/mol)",
        "ca_title": "3D Cα 蛋白質主鏈骨架軌跡 (Model {model})",
        "ca_ribbon_title": "3D 蛋白質二級結構飄帶模型 (Model {model})",
        "ca_colorbar": "殘基序號<br>(N → C 端)",
        "ca_legend": "多勝肽鏈 / 二級結構",
        "ca_chain_unit": "殘基",
        "ss_helix": "α-螺旋 (α-Helix)",
        "ss_sheet": "β-折板 (β-Sheet)",
        "ss_coil": "無規捲曲 / 環區 (Coil/Loop)",
        "cm_binary_title": "Cα 殘基接觸圖 — Model {model} Chain {chain} (閾值 ≤ {cutoff:.1f} Å)",
        "cm_dist_title": "Cα 殘基距離矩陣 — Model {model} Chain {chain}",
        "cm_sampled_tag": " [每 {step} 殘基取樣]",
        "cm_colorbar_contact": "空間接觸",
        "cm_colorbar_dist": "距離 (Å)",
        "cm_xaxis": "殘基序列編號 (Residue Index)",
        "cm_yaxis": "殘基序列編號 (Residue Index)",
        "cm_hover": "殘基 %{y} (%{customdata[0]}) ↔ 殘基 %{x} (%{customdata[1]})<br>Cα 距離: %{customdata[2]:.2f} Å<extra></extra>",
        "rama_title": "拉氏圖 (Ramachandran Plot) — Model {model} Chain {chain}",
        "rama_empty": "拉氏圖 — Model {model} Chain {chain} (無連續主鏈雙面角)",
        "rama_beta": "β-折板區 (β-Sheet)",
        "rama_alpha_r": "右手 α-螺旋區",
        "rama_alpha_l": "左手 α-螺旋",
        "rama_grp_gen": "一般胺基酸",
        "rama_grp_gly": "甘胺酸 (GLY)",
        "rama_grp_pro": "脯胺酸 (PRO)",
        "rama_xaxis": "φ (Phi) 雙面角 (度)",
        "rama_yaxis": "ψ (Psi) 雙面角 (度)",
        "rama_legend": "殘基類別",
        "kin_main_title": "阿瑞尼士／艾林動力學溫度掃描 (Arrhenius / Eyring Sweep)",
        "kin_sub_left": "速率常數隨溫度變化 (log₁₀ k)",
        "kin_sub_right": "阿瑞尼士線性圖 (ln k vs. 1000/T)",
        "kin_xaxis_t": "絕對溫度 T (K)",
        "kin_yaxis_logk": "log₁₀ k (s⁻¹)",
        "kin_xaxis_invt": "1000 / T (K⁻¹)",
        "kin_yaxis_lnk": "ln k"
    },
    "en": {
        "ca_gene_gc_colorbar": "Codon<br>GC%",
        "ca_gene_gc_title": "Model {model} 3D Structure–Codon GC% Co-translational Map",
        "nt_plot_title": "Deduced Coding cDNA Local GC% & Base Composition (Window = {win} nt)",
        "nt_sub_left": "Sliding-Window Local GC% Profile (5' → 3')",
        "nt_sub_right": "Nucleotide Base Composition",
        "nt_xaxis_pos": "Nucleotide Position (nt)",
        "nt_yaxis_gc": "Local GC Content (%)",
        "nt_yaxis_cnt": "Base Count (nt)",
        "energy_title": "Protein Folding Free Energy Landscape",
        "energy_xaxis": "Reaction Coordinate",
        "energy_yaxis": "Gibbs Free Energy (kJ/mol)",
        "energy_legend": "Models",
        "states_suffix": "States",
        "funnel_smooth": "Smooth Folding Funnel",
        "funnel_rugged": "Rugged Energy Landscape",
        "funnel_multi": "Multi-basin Energy Landscape",
        "funnel_x": "Coord 1",
        "funnel_y": "Coord 2",
        "funnel_z": "Effective Energy (kJ/mol)",
        "ca_title": "3D Cα Backbone Trace (Model {model})",
        "ca_ribbon_title": "3D Secondary Structure Ribbon Model (Model {model})",
        "ca_colorbar": "Residue Index<br>(N → C)",
        "ca_legend": "Chains / Secondary Structure",
        "ca_chain_unit": "aa",
        "ss_helix": "α-Helix",
        "ss_sheet": "β-Sheet",
        "ss_coil": "Coil / Loop",
        "cm_binary_title": "Cα Contact Map — Model {model} Chain {chain} (Cutoff ≤ {cutoff:.1f} Å)",
        "cm_dist_title": "Cα Distance Matrix — Model {model} Chain {chain}",
        "cm_sampled_tag": " [Stride: {step}]",
        "cm_colorbar_contact": "Contact",
        "cm_colorbar_dist": "Distance (Å)",
        "cm_xaxis": "Residue Index",
        "cm_yaxis": "Residue Index",
        "cm_hover": "Res %{y} (%{customdata[0]}) ↔ Res %{x} (%{customdata[1]})<br>Cα Distance: %{customdata[2]:.2f} Å<extra></extra>",
        "rama_title": "Ramachandran Plot — Model {model} Chain {chain}",
        "rama_empty": "Ramachandran Plot — Model {model} Chain {chain} (No continuous backbone angles)",
        "rama_beta": "β-Sheet Region",
        "rama_alpha_r": "Right-handed α-Helix",
        "rama_alpha_l": "Left-handed α",
        "rama_grp_gen": "General Residues",
        "rama_grp_gly": "Glycine (GLY)",
        "rama_grp_pro": "Proline (PRO)",
        "rama_xaxis": "φ (Phi) Angle (degrees)",
        "rama_yaxis": "ψ (Psi) Angle (degrees)",
        "rama_legend": "Residue Type",
        "kin_main_title": "Arrhenius / Eyring Temperature Sweep Kinetics",
        "kin_sub_left": "Rate Constants vs. Temperature (log₁₀ k)",
        "kin_sub_right": "Arrhenius Plot (ln k vs. 1000/T)",
        "kin_xaxis_t": "Temperature T (K)",
        "kin_yaxis_logk": "log₁₀ k (s⁻¹)",
        "kin_xaxis_invt": "1000 / T (K⁻¹)",
        "kin_yaxis_lnk": "ln k"
    }
}


def _get_pt(lang):
    return PLOT_TEXTS.get(lang, PLOT_TEXTS["en"])


# =============================
# 平滑多尺度隨機地形 (2D FFT 光譜合成)
# =============================
def generate_irregular_field(size, cutoff_freq, rng):
    noise = rng.normal(0, 1, (size, size))
    fft_noise = np.fft.fft2(noise)

    freqs = np.fft.fftfreq(size) * size
    fx, fy = np.meshgrid(freqs, freqs)
    freq_sq = fx**2 + fy**2

    filter_kernel = np.exp(-freq_sq / (2.0 * cutoff_freq**2))
    field = np.real(np.fft.ifft2(fft_noise * filter_kernel))

    field -= np.mean(field)
    field /= np.std(field) + 1e-8
    return field


# =====================================
# 1. 2D Free Energy Comparison
# =====================================
def create_energy_figure(model_a, model_b, lang="en"):
    pt = _get_pt(lang)
    fig = go.Figure()
    models = [("Model A", model_a), ("Model B", model_b)]

    for name, model in models:
        fig.add_trace(
            go.Scatter(
                x=model["x"],
                y=model["y"],
                mode="lines",
                name=name,
                line=dict(width=3)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 0.5, 1],
                y=[model["g_unfolded"], model["g_transition"], model["g_native"]],
                mode="markers",
                name=f"{name} {pt['states_suffix']}",
                marker=dict(size=9),
                showlegend=False
            )
        )

    fig.update_layout(
        title=pt["energy_title"],
        xaxis_title=pt["energy_xaxis"],
        yaxis_title=pt["energy_yaxis"],
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend_title=pt["energy_legend"]
    )
    return fig


# =====================================
# 2. 3D Folding Funnel
# =====================================
def create_funnel_figure(
    model,
    funnel_type="smooth",
    roughness_amplitude=2.5,
    basin_depth=9.0,
    basin_width=0.08,
    barrier_scale=0.15,
    funnel_width_scale=1.2,
    seed=42,
    z_range=None,
    model_label=None,
    height=720,
    lang="en"
):
    pt = _get_pt(lang)
    size = 240
    axis = np.linspace(-1, 1, size)
    U, V = np.meshgrid(axis, axis)
    R = np.sqrt(U**2 + V**2)
    outside = R > 1.0

    g_unfolded = model["g_unfolded"]
    g_transition = model["g_transition"]
    g_native = model["g_native"]

    R_clipped = np.clip(R, 0.0, 1.0)
    base_profile = g_native + (g_unfolded - g_native) * (R_clipped ** 1.4)

    shoulder_height = max(0.0, g_transition - g_unfolded) * barrier_scale
    shoulder = shoulder_height * np.exp(-((R_clipped - 0.55) / 0.14)**2)

    Z_base = base_profile + shoulder

    if funnel_type == "smooth":
        Z = Z_base

    elif funnel_type == "rugged":
        rng = np.random.default_rng(seed)

        large_scale = generate_irregular_field(size, cutoff_freq=4.0, rng=rng)
        medium_scale = generate_irregular_field(size, cutoff_freq=10.0, rng=rng)
        fine_scale = generate_irregular_field(size, cutoff_freq=22.0, rng=rng)

        terrain = (
            0.45 * large_scale
            + 0.35 * medium_scale
            + 0.20 * fine_scale
        )

        local_features = np.zeros_like(U)
        for _ in range(35):
            r_pos = rng.uniform(0.22, 0.85)
            theta_pos = rng.uniform(0, 2 * np.pi)
            cx = r_pos * np.cos(theta_pos)
            cy = r_pos * np.sin(theta_pos)

            angle = rng.uniform(0, 2 * np.pi)
            major = rng.uniform(0.05, 0.14)
            minor = rng.uniform(0.03, 0.07)

            dx = U - cx
            dy = V - cy
            xr = dx * np.cos(angle) + dy * np.sin(angle)
            yr = -dx * np.sin(angle) + dy * np.cos(angle)

            strength = rng.uniform(-1.8, 1.2)
            feature = strength * np.exp(-0.5 * ((xr / major)**2 + (yr / minor)**2))
            local_features += feature

        raw_roughness = terrain + 0.7 * local_features

        envelope = (R_clipped / 0.55)**2 * np.exp(1.0 - (R_clipped / 0.55)**2)
        envelope = np.clip(envelope, 0.0, 1.1)

        Z_rugged = Z_base + roughness_amplitude * envelope * raw_roughness

        delta_g = max(2.0, abs(g_unfolded - g_native))
        min_gap = 0.15 * delta_g
        floor_curve = (
            g_native
            - 5.0 * np.exp(-(R_clipped / 0.15)**2)
            + min_gap * (1.0 - np.exp(-(R_clipped / 0.25)**2))
        )
        k = 5.0
        Z = np.logaddexp(k * Z_rugged, k * floor_curve) / k

    elif funnel_type == "multi":
        basin_1 = -basin_depth * np.exp(-((U + 0.35)**2 / basin_width + (V - 0.20)**2 / basin_width))
        basin_2 = -basin_depth * np.exp(-((U - 0.25)**2 / basin_width + (V + 0.40)**2 / basin_width))
        basin_3 = -(basin_depth * 0.7) * np.exp(-((U - 0.55)**2 / basin_width + (V - 0.15)**2 / basin_width))
        Z = Z_base + basin_1 + basin_2 + basin_3

    else:
        raise ValueError("Unknown funnel type")

    X_plot = U * funnel_width_scale
    Y_plot = V * funnel_width_scale
    Z = np.where(outside, np.nan, Z)

    titles = {
        "smooth": pt["funnel_smooth"],
        "rugged": pt["funnel_rugged"],
        "multi": pt["funnel_multi"]
    }
    chart_title = f"{model_label} — {titles[funnel_type]}" if model_label else titles[funnel_type]

    surface_kwargs = dict(
        x=X_plot,
        y=Y_plot,
        z=Z,
        colorscale="Viridis",
        colorbar=dict(title="kJ/mol"),
        connectgaps=False
    )
    if z_range is not None:
        surface_kwargs["cmin"] = z_range[0]
        surface_kwargs["cmax"] = z_range[1]

    fig = go.Figure(data=[go.Surface(**surface_kwargs)])

    zaxis_cfg = dict(title=pt["funnel_z"])
    if z_range is not None:
        zaxis_cfg["range"] = z_range

    fig.update_layout(
        title=chart_title,
        scene=dict(
            xaxis_title=pt["funnel_x"],
            yaxis_title=pt["funnel_y"],
            zaxis=zaxis_cfg,
            aspectratio=dict(
                x=funnel_width_scale,
                y=funnel_width_scale,
                z=0.8
            )
        ),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def _catmull_rom_spline(coords, subdivisions=6):
    """對 3D 主鏈座標進行 Catmull-Rom 樣條平滑內插，產生連續流暢的飄帶曲線"""
    n = len(coords)
    if n < 3:
        return coords, np.arange(n)

    padded = np.vstack([2 * coords[0] - coords[1], coords, 2 * coords[-1] - coords[-2]])
    smooth_pts = []
    parent_idx = []

    t = np.linspace(0, 1, subdivisions, endpoint=False)
    t2 = t * t
    t3 = t2 * t

    for i in range(n - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        smooth_pts.append(seg)
        half = subdivisions // 2
        parent_idx.extend([i] * half + [i + 1] * (subdivisions - half))

    smooth_pts.append(coords[-1][None, :])
    parent_idx.append(n - 1)
    return np.vstack(smooth_pts), np.array(parent_idx)


# =====================================
# 3. 3D C-alpha Backbone & Secondary Structure Ribbon Model
# =====================================
def create_ca_backbone_figure(
    chain_results,
    selected_model=0,
    color_mode="ribbon",
    show_markers=False,
    height=660,
    lang="en"
):
    pt = _get_pt(lang)
    fig = go.Figure()

    target_chains = [item for item in chain_results if item["model"] == selected_model]

    ss_styles = {
        "H": {"name": pt["ss_helix"], "color": "#E11D48", "width": 11},
        "E": {"name": pt["ss_sheet"], "color": "#2563EB", "width": 9},
        "C": {"name": pt["ss_coil"],  "color": "#94A3B8", "width": 3.5},
    }
    legend_shown = {"H": False, "E": False, "C": False}

    for item in target_chains:
        coords = np.array(item["ca_coords"])
        if len(coords) == 0:
            continue

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        labels = item["res_labels"]
        chain_id = item["chain"]
        n_res = len(coords)
        ss_list = item.get("ss_list", ["C"] * n_res)

        if color_mode == "ribbon":
            smooth_coords, parent_idx = _catmull_rom_spline(coords, subdivisions=6)
            smooth_ss = [ss_list[idx] for idx in parent_idx]

            m = len(smooth_coords)
            seg_start = 0
            while seg_start < m:
                curr_state = smooth_ss[seg_start]
                seg_end = seg_start + 1
                while seg_end < m and smooth_ss[seg_end] == curr_state:
                    seg_end += 1

                draw_end = min(m, seg_end + 1)
                seg_pts = smooth_coords[seg_start:draw_end]
                style = ss_styles[curr_state]

                fig.add_trace(
                    go.Scatter3d(
                        x=seg_pts[:, 0],
                        y=seg_pts[:, 1],
                        z=seg_pts[:, 2],
                        mode="lines",
                        name=style["name"],
                        legendgroup=curr_state,
                        showlegend=not legend_shown[curr_state],
                        line=dict(color=style["color"], width=style["width"]),
                        hoverinfo="skip"
                    )
                )
                legend_shown[curr_state] = True
                seg_start = seg_end

            node_colors = [ss_styles[s]["color"] for s in ss_list]
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    name=f"Chain {chain_id} Cα",
                    text=labels,
                    hoverinfo="text+x+y+z",
                    showlegend=False,
                    marker=dict(
                        size=3.8 if show_markers else 2.5,
                        color=node_colors,
                        opacity=0.95 if show_markers else 0.01
                    )
                )
            )

        else:
            mode = "lines+markers" if show_markers else "lines"
            if color_mode == "sequence":
                seq_idx = np.arange(1, n_res + 1)
                line_cfg = dict(
                    width=5,
                    color=seq_idx,
                    colorscale="Turbo",
                    colorbar=dict(
                        title=dict(text=pt["ca_colorbar"], side="top"),
                        orientation="v",
                        x=1.02,
                        xanchor="left",
                        y=0.35,
                        yanchor="middle",
                        len=0.55,
                        thickness=14
                    ) if len(target_chains) == 1 else None
                )
                marker_cfg = dict(
                    size=3.5,
                    color=seq_idx,
                    colorscale="Turbo"
                )
            elif color_mode == "codon_gc":
                gc_vals = item.get("codon_gc_list", [50.0] * n_res)
                line_cfg = dict(
                    width=6,
                    color=gc_vals,
                    colorscale="RdYlBu_r",
                    cmin=0,
                    cmax=100,
                    colorbar=dict(
                        title=dict(text=pt["ca_gene_gc_colorbar"], side="top"),
                        orientation="v",
                        x=1.02,
                        xanchor="left",
                        y=0.35,
                        yanchor="middle",
                        len=0.55,
                        thickness=14
                    ) if len(target_chains) == 1 else None
                )
                marker_cfg = dict(
                    size=4.0,
                    color=gc_vals,
                    colorscale="RdYlBu_r",
                    cmin=0,
                    cmax=100
                )
            else:
                line_cfg = dict(width=5)
                marker_cfg = dict(size=3.5)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode=mode,
                    name=f"Chain {chain_id} ({n_res} {pt['ca_chain_unit']})",
                    text=labels,
                    hoverinfo="text+x+y+z",
                    line=line_cfg,
                    marker=marker_cfg
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=[x[0], x[-1]],
                y=[y[0], y[-1]],
                z=[z[0], z[-1]],
                mode="markers+text",
                text=[f"{chain_id}:N", f"{chain_id}:C"],
                textposition="top center",
                marker=dict(size=6, symbol="diamond", color="#0F172A"),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    if color_mode == "ribbon":
        title_str = pt["ca_ribbon_title"].format(model=selected_model)
    elif color_mode == "codon_gc":
        title_str = pt["ca_gene_gc_title"].format(model=selected_model)
    else:
        title_str = pt["ca_title"].format(model=selected_model)

    fig.update_layout(
        title=dict(text=title_str, x=0.02, y=0.97),
        autosize=True,
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data"
        ),
        height=height,
        margin=dict(l=10, r=90, t=55, b=15),
        legend=dict(
            x=1.02,
            xanchor="left",
            y=0.98,
            yanchor="top"
        )
    )

    return fig


# =====================================
# 4. Residue-Residue Contact Map (向量化高速渲染 + 大型結構保護)
# =====================================
# =====================================
# 4. Residue-Residue Contact Map (向量化高速渲染 + 大型結構保護)
# =====================================
# =====================================
# 4. Residue-Residue Contact Map (純原生列表 + 向量化 hovertemplate)
# =====================================
def create_contact_map_figure(
    chain_item,
    cutoff=8.0,
    map_mode="binary",
    height=620,
    lang="en",
    max_display_residues=1000
):
    pt = _get_pt(lang)
    coords = np.array(chain_item["ca_coords"], dtype=float)
    chain_id = chain_item["chain"]
    model_id = chain_item["model"]
    seq = chain_item.get("sequence", "")
    n_total = len(coords)

    if n_total == 0:
        return go.Figure()

    step = max(1, int(np.ceil(n_total / max_display_residues)))
    coords_sub = coords[::step]
    n_sub = len(coords_sub)

    res_indices = np.arange(1, n_total + 1, step)[:n_sub]

    # 建構帶有胺基酸代碼的座標標籤 (例如 "1 (M)", "2 (Q)")，避免依賴 customdata
    axis_labels = [
        f"{idx} ({seq[idx - 1] if (idx - 1) < len(seq) else '?'})"
        for idx in res_indices
    ]

    diff = coords_sub[:, np.newaxis, :] - coords_sub[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    sampled_suffix = pt["cm_sampled_tag"].format(step=step) if step > 1 else ""

    if map_mode == "binary":
        z_data = (dist_matrix <= cutoff).astype(int).tolist()
        colorscale = [[0.0, "#F8FAFC"], [1.0, "#1E3A8A"]]
        colorbar_cfg = dict(
            title=pt["cm_colorbar_contact"],
            tickvals=[0, 1],
            ticktext=[f"> {cutoff:.1f} Å", f"≤ {cutoff:.1f} Å"]
        )
        title_text = pt["cm_binary_title"].format(
            model=model_id, chain=chain_id, cutoff=cutoff
        ) + sampled_suffix
        hover_tmpl = (
            "殘基 Y: %{y} ↔ 殘基 X: %{x}<br>接觸狀態: %{z}<extra></extra>"
            if lang == "zh"
            else "Res Y: %{y} ↔ Res X: %{x}<br>Contact: %{z}<extra></extra>"
        )
    else:
        z_data = np.round(dist_matrix, 2).tolist()
        colorscale = "Viridis_r"
        colorbar_cfg = dict(title=pt["cm_colorbar_dist"])
        title_text = pt["cm_dist_title"].format(
            model=model_id, chain=chain_id
        ) + sampled_suffix
        hover_tmpl = (
            "殘基 Y: %{y} ↔ 殘基 X: %{x}<br>Cα 距離: %{z:.2f} Å<extra></extra>"
            if lang == "zh"
            else "Res Y: %{y} ↔ Res X: %{x}<br>Cα Distance: %{z:.2f} Å<extra></extra>"
        )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_data,
                x=axis_labels,
                y=axis_labels,
                hovertemplate=hover_tmpl,
                colorscale=colorscale,
                colorbar=colorbar_cfg,
                showscale=True
            )
        ]
    )

    fig.update_layout(
        title=title_text,
        xaxis=dict(title=pt["cm_xaxis"], constrain="domain", nticks=15),
        yaxis=dict(title=pt["cm_yaxis"], scaleanchor="x", scaleratio=1, nticks=15),
        height=height,
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig


# =====================================
# 5. Ramachandran Plot (Phi / Psi)
# =====================================
def create_ramachandran_figure(chain_item, height=620, lang="en"):
    pt = _get_pt(lang)
    rama_data = chain_item.get("rama_data", [])
    chain_id = chain_item["chain"]
    model_id = chain_item["model"]

    fig = go.Figure()

    if not rama_data:
        fig.update_layout(
            title=pt["rama_empty"].format(model=model_id, chain=chain_id),
            height=height
        )
        return fig

    fig.add_shape(
        type="rect", x0=-165, y0=90, x1=-45, y1=175,
        fillcolor="rgba(59, 130, 246, 0.14)",
        line=dict(color="rgba(59, 130, 246, 0.45)", width=1, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=-105, y=135, text=pt["rama_beta"],
        showarrow=False, font=dict(size=11, color="#1D4ED8")
    )

    fig.add_shape(
        type="rect", x0=-110, y0=-70, x1=-35, y1=-15,
        fillcolor="rgba(16, 185, 129, 0.16)",
        line=dict(color="rgba(16, 185, 129, 0.55)", width=1, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=-72, y=-42, text=pt["rama_alpha_r"],
        showarrow=False, font=dict(size=11, color="#047857")
    )

    fig.add_shape(
        type="rect", x0=30, y0=10, x1=90, y1=75,
        fillcolor="rgba(245, 158, 11, 0.15)",
        line=dict(color="rgba(245, 158, 11, 0.50)", width=1, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=60, y=42, text=pt["rama_alpha_l"],
        showarrow=False, font=dict(size=10, color="#B45309")
    )

    fig.add_hline(y=0, line_width=1, line_color="#CBD5E1")
    fig.add_vline(x=0, line_width=1, line_color="#CBD5E1")

    groups = [
        (pt["rama_grp_gen"], [d for d in rama_data if d["res_name"] not in ("GLY", "PRO")], "circle", "#2563EB"),
        (pt["rama_grp_gly"], [d for d in rama_data if d["res_name"] == "GLY"], "diamond", "#DC2626"),
        (pt["rama_grp_pro"], [d for d in rama_data if d["res_name"] == "PRO"], "square", "#7C3AED"),
    ]

    for group_name, items, symbol, color in groups:
        if not items:
            continue
        fig.add_trace(
            go.Scatter(
                x=[d["phi"] for d in items],
                y=[d["psi"] for d in items],
                mode="markers",
                name=f"{group_name} ({len(items)})",
                text=[
                    f"{d['res_name']} {d['res_num']} ({d['aa_code']})<br>"
                    f"φ (Phi): {d['phi']:.1f}°<br>"
                    f"ψ (Psi): {d['psi']:.1f}°"
                    for d in items
                ],
                hoverinfo="text",
                marker=dict(
                    size=8,
                    symbol=symbol,
                    color=color,
                    opacity=0.85,
                    line=dict(width=0.8, color="white")
                )
            )
        )

    fig.update_layout(
        title=pt["rama_title"].format(model=model_id, chain=chain_id),
        xaxis=dict(
            title=pt["rama_xaxis"],
            range=[-180, 180],
            dtick=60,
            constrain="domain"
        ),
        yaxis=dict(
            title=pt["rama_yaxis"],
            range=[-180, 180],
            dtick=60,
            scaleanchor="x",
            scaleratio=1
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(title=pt["rama_legend"])
    )

    return fig


# =====================================
# 6. Arrhenius / Eyring Temperature Sweep
# =====================================
def create_kinetics_sweep_figure(sweep_a, sweep_b, height=560, lang="en"):
    pt = _get_pt(lang)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(pt["kin_sub_left"], pt["kin_sub_right"]),
        horizontal_spacing=0.10
    )

    fig.add_trace(
        go.Scatter(
            x=sweep_a["T"],
            y=np.log10(sweep_a["k_fold"]),
            mode="lines",
            name="Model A: k_fold",
            line=dict(color="#2563EB", width=3)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_a["T"],
            y=np.log10(sweep_a["k_unfold"]),
            mode="lines",
            name="Model A: k_unfold",
            line=dict(color="#2563EB", width=2.5, dash="dash")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sweep_b["T"],
            y=np.log10(sweep_b["k_fold"]),
            mode="lines",
            name="Model B: k_fold",
            line=dict(color="#EA580C", width=3)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_b["T"],
            y=np.log10(sweep_b["k_unfold"]),
            mode="lines",
            name="Model B: k_unfold",
            line=dict(color="#EA580C", width=2.5, dash="dash")
        ),
        row=1, col=1
    )

    if sweep_a["t_m"] is not None:
        fig.add_vline(
            x=sweep_a["t_m"], line_dash="dot", line_color="#2563EB",
            annotation_text=f"Tm(A)≈{sweep_a['t_m']:.1f}K",
            annotation_position="top left",
            row=1, col=1
        )
    if sweep_b["t_m"] is not None:
        fig.add_vline(
            x=sweep_b["t_m"], line_dash="dot", line_color="#EA580C",
            annotation_text=f"Tm(B)≈{sweep_b['t_m']:.1f}K",
            annotation_position="bottom right",
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=sweep_a["inv_T"],
            y=sweep_a["ln_k_fold"],
            mode="lines",
            name="Model A: ln(k_fold)",
            line=dict(color="#2563EB", width=3),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_a["inv_T"],
            y=sweep_a["ln_k_unfold"],
            mode="lines",
            name="Model A: ln(k_unfold)",
            line=dict(color="#2563EB", width=2.5, dash="dash"),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_b["inv_T"],
            y=sweep_b["ln_k_fold"],
            mode="lines",
            name="Model B: ln(k_fold)",
            line=dict(color="#EA580C", width=3),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_b["inv_T"],
            y=sweep_b["ln_k_unfold"],
            mode="lines",
            name="Model B: ln(k_unfold)",
            line=dict(color="#EA580C", width=2.5, dash="dash"),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text=pt["kin_xaxis_t"], row=1, col=1)
    fig.update_yaxes(title_text=pt["kin_yaxis_logk"], row=1, col=1)
    fig.update_xaxes(title_text=pt["kin_xaxis_invt"], row=1, col=2)
    fig.update_yaxes(title_text=pt["kin_yaxis_lnk"], row=1, col=2)

    fig.update_layout(
        title=pt["kin_main_title"],
        template="plotly_white",
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5)
    )

    return fig


# =====================================
# 7. Nucleotide Sequence Local GC% & Base Composition
# =====================================
def create_nucleotide_analysis_figure(cdna_seq, window=15, height=430, lang="en"):
    pt = _get_pt(lang)
    seq = cdna_seq.upper()
    n = len(seq)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=(pt["nt_sub_left"], pt["nt_sub_right"]),
        horizontal_spacing=0.10
    )

    if n == 0:
        return fig

    eff_win = min(window, max(3, n))
    positions = []
    local_gc = []

    for i in range(n - eff_win + 1):
        sub = seq[i : i + eff_win]
        gc_score = (
            sub.count("G") + sub.count("C") + sub.count("S")
            + 0.5 * sum(sub.count(b) for b in ("R", "Y", "K", "M", "N"))
        )
        positions.append(int(i + 1 + eff_win // 2))
        local_gc.append(round(gc_score / eff_win * 100.0, 2))

    mean_gc = float(np.mean(local_gc)) if local_gc else 0.0

    fig.add_trace(
        go.Scatter(
            x=positions,
            y=local_gc,
            mode="lines",
            name="Local GC%",
            line=dict(color="#0D9488", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(13, 148, 136, 0.15)",
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_hline(
        y=mean_gc,
        line_dash="dash",
        line_color="#0F766E",
        annotation_text=f"Mean: {mean_gc:.1f}%",
        annotation_position="top right",
        row=1, col=1
    )
    fig.add_hline(y=50.0, line_dash="dot", line_color="#94A3B8", row=1, col=1)

    bases = ["A", "T", "G", "C"]
    counts = [int(seq.count(b)) for b in bases]
    colors = ["#3B82F6", "#F59E0B", "#10B981", "#6366F1"]

    fig.add_trace(
        go.Bar(
            x=bases,
            y=counts,
            marker_color=colors,
            text=[f"{c}<br>({c / n * 100:.1f}%)" for c in counts],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text=pt["nt_xaxis_pos"], row=1, col=1)
    fig.update_yaxes(title_text=pt["nt_yaxis_gc"], range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text=pt["nt_yaxis_cnt"], row=1, col=2)

    fig.update_layout(
        title=pt["nt_plot_title"].format(win=eff_win),
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=25)
    )

    return fig


# =====================================
# 8. 3Dmol.js 全功能 3D 分子檢視器嵌入 HTML
# =====================================
def create_3dmol_html(
    pdb_text,
    height=600,
    default_style="cartoon",
    default_color="spectrum",
    show_sidechains=False,
    surface_mode="none",
    spin=False
):
    """
    產生包含 3Dmol.js 引擎、頂部互動工具列與深色現代畫布的完整 HTML 字串。
    支援 Cartoon / Sticks / Spheres / Surface / 側鏈以及懸停資訊。
    """
    # 避免跳脫問題，將 PDB 文字安全存於 <script type="text/plain"> 標籤
    html_code = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0B1120; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; }}
    #container {{ width: 100%; height: {height}px; position: relative; border-radius: 12px; overflow: hidden; }}
    #viewport {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}
    #toolbar {{
      position: absolute; top: 12px; left: 12px; right: 12px;
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
      background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(10px);
      padding: 8px 14px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);
      z-index: 10; color: #E2E8F0; font-size: 13px;
    }}
    .ctrl-group {{ display: flex; align-items: center; gap: 5px; }}
    label {{ color: #94A3B8; font-weight: 500; font-size: 12px; }}
    select, button {{
      background: #1E293B; color: #F8FAFC; border: 1px solid #334155;
      padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; outline: none;
      transition: all 0.2s;
    }}
    select:hover, button:hover {{ background: #334155; border-color: #64748B; }}
    .badge {{
      margin-left: auto; font-size: 11px; padding: 3px 8px; border-radius: 20px;
      background: rgba(13, 148, 136, 0.25); color: #2DD4BF; border: 1px solid rgba(45, 212, 191, 0.3);
    }}
    #tooltip {{
      position: absolute; bottom: 12px; left: 12px;
      background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px);
      color: #38BDF8; padding: 6px 12px; border-radius: 8px;
      border: 1px solid rgba(56, 189, 248, 0.3); font-size: 12px; pointer-events: none;
      display: none; z-index: 10;
    }}
  </style>
</head>
<body>
  <div id="container">
    <div id="toolbar">
      <div class="ctrl-group">
        <label>樣式:</label>
        <select id="style_select" onchange="updateViewer()">
          <option value="cartoon" selected>立體卡通 (Cartoon)</option>
          <option value="stick">棒狀模型 (Sticks)</option>
          <option value="sphere">空間填充球 (Spheres)</option>
          <option value="line">細線骨架 (Lines)</option>
        </select>
      </div>
      <div class="ctrl-group">
        <label>著色:</label>
        <select id="color_select" onchange="updateViewer()">
          <option value="spectrum" selected>光譜彩虹 (Spectrum)</option>
          <option value="ss">二級結構 (Helix/Sheet/Coil)</option>
          <option value="chain">多勝肽鏈 (Chain)</option>
          <option value="hydrophobicity">疏水性色階 (Hydrophobic)</option>
        </select>
      </div>
      <div class="ctrl-group">
        <label>表面:</label>
        <select id="surface_select" onchange="updateViewer()">
          <option value="none" selected>無 (None)</option>
          <option value="sas">溶劑可及表面 (SAS)</option>
          <option value="vdw">凡得瓦表面 (VDW)</option>
        </select>
      </div>
      <div class="ctrl-group">
        <label style="cursor:pointer;"><input type="checkbox" id="chk_sidechains" onchange="updateViewer()"> 側鏈</label>
      </div>
      <div class="ctrl-group">
        <label style="cursor:pointer;"><input type="checkbox" id="chk_spin" onchange="toggleSpin()"> 自轉</label>
      </div>
      <button onclick="resetCamera()">🔄 重置視角</button>
      <span class="badge">3Dmol.js WebGL</span>
    </div>

    <div id="viewport"></div>
    <div id="tooltip"></div>
  </div>

  <script type="text/plain" id="raw_pdb">{pdb_text}</script>

  <script>
    let viewer = null;
    let currentSurface = null;
    let isSpinning = false;

    window.addEventListener('DOMContentLoaded', () => {{
      let element = document.getElementById('viewport');
      let config = {{ backgroundColor: '#0B1120' }};
      viewer = $3Dmol.createViewer(element, config);

      let pdbData = document.getElementById('raw_pdb').textContent;
      viewer.addModel(pdbData, "pdb");
      updateViewer();
      viewer.zoomTo();
      viewer.render();

      // 懸停殘基資訊提示
      viewer.setHoverable({{}}, true,
        function(atom, viewer, event, container) {{
          if (atom) {{
            let tip = document.getElementById('tooltip');
            tip.style.display = 'block';
            tip.innerHTML = `<strong>${{atom.resn}} ${{atom.resi}}</strong> (鏈 ${{atom.chain}}) - 原子: ${{atom.atom}}`;
          }}
        }},
        function(atom, viewer, event, container) {{
          document.getElementById('tooltip').style.display = 'none';
        }}
      );
    }});

    function updateViewer() {{
      if (!viewer) return;
      let styleMode = document.getElementById('style_select').value;
      let colorMode = document.getElementById('color_select').value;
      let surfaceMode = document.getElementById('surface_select').value;
      let showSidechains = document.getElementById('chk_sidechains').checked;

      let colorScheme = {{}};
      if (colorMode === 'spectrum') {{
        colorScheme = {{ color: 'spectrum' }};
      }} else if (colorMode === 'ss') {{
        colorScheme = {{ colorscheme: 'ssJmol' }};
      }} else if (colorMode === 'chain') {{
        colorScheme = {{ colorscheme: 'chain' }};
      }} else if (colorMode === 'hydrophobicity') {{
        colorScheme = {{ colorscheme: 'amino' }};
      }}

      let baseStyle = Object.assign({{}}, colorScheme);

      let mainStyle = {{}};
      if (styleMode === 'cartoon') {{
        mainStyle.cartoon = baseStyle;
      }} else if (styleMode === 'stick') {{
        mainStyle.stick = baseStyle;
      }} else if (styleMode === 'sphere') {{
        mainStyle.sphere = Object.assign({{ scale: 0.35 }}, baseStyle);
      }} else if (styleMode === 'line') {{
        mainStyle.line = baseStyle;
      }}

      viewer.setStyle({{}}, mainStyle);

      if (showSidechains && styleMode === 'cartoon') {{
        viewer.setStyle({{ hetflag: false }}, {{
          cartoon: baseStyle,
          stick: {{ radius: 0.16, colorscheme: 'amino' }}
        }});
      }}

      // 配體/異質分子 (HETATM) 永遠以棒狀加空間填充凸顯
      viewer.setStyle({{ hetflag: true }}, {{
        stick: {{ radius: 0.28, colorscheme: 'element' }},
        sphere: {{ scale: 0.3 }}
      }});

      // 表面渲染
      if (currentSurface) {{
        viewer.removeSurface(currentSurface);
        currentSurface = null;
      }}

      if (surfaceMode === 'sas') {{
        currentSurface = viewer.addSurface($3Dmol.SurfaceType.SAS, {{
          opacity: 0.65,
          color: colorMode === 'spectrum' ? 'spectrum' : '#38BDF8'
        }});
      }} else if (surfaceMode === 'vdw') {{
        currentSurface = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
          opacity: 0.55,
          color: '#818CF8'
        }});
      }}

      viewer.render();
    }}

    function toggleSpin() {{
      if (!viewer) return;
      isSpinning = !isSpinning;
      viewer.spin(isSpinning, [0, 1, 0]);
    }}

    function resetCamera() {{
      if (!viewer) return;
      viewer.zoomTo();
      viewer.render();
    }}
  </script>
</body>
</html>
"""
    return html_code


# =====================================
# 9. 獨立可儲存/列印之完整研究與實驗報告 HTML 產生器
# =====================================
def generate_standalone_report_html(data_bundle):
    """
    將所有結構、熱力學、二硫鍵、突變、引子與生化指標編譯為
    自帶現代 CSS 樣式的單一獨立 HTML 檔案，使用者可直接存檔或列印為 PDF。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = data_bundle.get("title", "Protein Thermodynamics Report")
    seq = data_bundle.get("sequence", "")
    l = len(seq)
    bp = data_bundle.get("bp_stats", {})
    bio = data_bundle.get("bio_stats", {})
    nt = data_bundle.get("nt_stats", {})
    primers = data_bundle.get("primers", {})
    thermo = data_bundle.get("thermo_stats", {})
    ss_comp = data_bundle.get("ss_comp", {})
    ss_candidates = data_bundle.get("ss_candidates", [])
    mut_info = data_bundle.get("mut_info", None)

    # 建立二硫鍵候選表格 HTML
    ss_rows = ""
    for c in ss_candidates[:6]:
        ss_rows += f"""<tr>
          <td><strong>{c['res1_aa']}{c['res1_num']} ({c['res1_ss']})</strong></td>
          <td><strong>{c['res2_aa']}{c['res2_num']} ({c['res2_ss']})</strong></td>
          <td>{c['d_ca']:.2f} Å</td>
          <td>{c['d_cb']:.2f} Å</td>
          <td>{c['seq_dist']} aa</td>
          <td style="color:#059669;font-weight:600;">{c['est_ddg_kj']} kJ/mol</td>
          <td style="color:#2563EB;font-weight:600;">+{c['est_tm_boost']} °C</td>
        </tr>"""
    if not ss_rows:
        ss_rows = "<tr><td colspan='7' style='text-align:center;color:#94A3B8;'>無符合幾何標準之工程位點</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <title>{title} — 分析與實驗設計報告</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --primary: #0F172A;
      --accent: #0D9488;
      --accent-light: #F0FDFA;
      --text: #1E293B;
      --muted: #64748B;
      --card-bg: #FFFFFF;
      --border: #E2E8F0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: #F8FAFC; color: var(--text); line-height: 1.6; padding: 30px 20px;
    }}
    .container {{ max-width: 1080px; margin: 0 auto; }}
    .header {{
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
      color: #FFFFFF; padding: 32px 36px; border-radius: 16px; margin-bottom: 24px;
      box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }}
    .header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px; }}
    .header p {{ color: #94A3B8; font-size: 14px; }}
    .btn-print {{
      display: inline-block; margin-top: 16px; background: #0D9488; color: white;
      padding: 8px 18px; border-radius: 8px; text-decoration: none; font-size: 13px; font-weight: 600;
      cursor: pointer; border: none; transition: background 0.2s;
    }}
    .btn-print:hover {{ background: #0F766E; }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .stat-card {{
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px;
      padding: 18px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    .stat-card .label {{ font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; }}
    .stat-card .value {{ font-size: 22px; font-weight: 700; color: var(--primary); margin-top: 4px; }}
    .card {{
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px;
      padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
    }}
    .card h2 {{
      font-size: 18px; font-weight: 700; color: var(--primary); margin-bottom: 16px;
      padding-bottom: 10px; border-bottom: 2px solid var(--accent-light);
      display: flex; align-items: center; gap: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13.5px; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: #F1F5F9; color: var(--muted); font-weight: 600; font-size: 12px; }}
    tr:hover td {{ background: #F8FAFC; }}
    .seq-box {{
      background: #0B1120; color: #38BDF8; padding: 16px; border-radius: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px; word-break: break-all; line-height: 1.5; margin-top: 10px;
    }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
    .badge-success {{ background: #DCFCE7; color: #166534; }}
    .badge-warning {{ background: #FEF3C7; color: #92400E; }}
    .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 36px; }}
    @media print {{
      body {{ background: white; padding: 0; }}
      .btn-print {{ display: none; }}
      .card, .header {{ box-shadow: none; border: 1px solid #CBD5E1; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🧬 蛋白質生物物理與分子工程綜合分析報告</h1>
      <p>目標標的：{title} ｜ 產生時間：{now_str} ｜ 鏈長度：{l} 殘基 (aa)</p>
      <button class="btn-print" onclick="window.print()">🖨️ 列印 / 另存為 PDF</button>
    </div>

    <div class="grid-4">
      <div class="stat-card">
        <div class="label">拓撲折疊架構</div>
        <div class="value">{ss_comp.get('fold_class', 'α/β Class')}</div>
      </div>
      <div class="stat-card">
        <div class="label">相對接觸級數 (RCO)</div>
        <div class="value">{bp.get('rco', 0.0):.3f}</div>
      </div>
      <div class="stat-card">
        <div class="label">分子量 / 等電點</div>
        <div class="value">{bio.get('mw_kda', 0.0):.2f} kDa <span style="font-size:15px;color:var(--muted);">(pI {bio.get('pi', 0.0):.1f})</span></div>
      </div>
      <div class="stat-card">
        <div class="label">熱熔解溫度 Tm</div>
        <div class="value">{thermo.get('tm_str', 'N/A')}</div>
      </div>
    </div>

    <!-- 區塊 1：結構生物物理與二級結構 -->
    <div class="card">
      <h2>📊 1. 結構拓撲與生物物理指標診斷</h2>
      <table>
        <tr>
          <th>迴轉半徑 (Rg)</th>
          <td>{bp.get('rg', 0.0):.2f} Å</td>
          <th>長程三級接觸佔比</th>
          <td>{bp.get('long_range_ratio', 0.0)*100:.1f}%</td>
        </tr>
        <tr>
          <th>疏水核心埋藏率</th>
          <td>{bp.get('core_burial_ratio', 0.0)*100:.1f}%</td>
          <th>非相鄰總接觸數</th>
          <td>{bp.get('n_contacts', 0)} 對</td>
        </tr>
        <tr>
          <th>二級結構組成</th>
          <td colspan="3">α-螺旋: <strong>{ss_comp.get('h_pct', 0.0):.1f}%</strong> ｜ β-折板: <strong>{ss_comp.get('e_pct', 0.0):.1f}%</strong> ｜ 環區/無規捲曲: <strong>{ss_comp.get('c_pct', 0.0):.1f}%</strong></td>
        </tr>
      </table>
    </div>

    <!-- 區塊 2：二硫鍵與突變工程 -->
    <div class="card">
      <h2>🔬 2. 蛋白質工程：二硫鍵與穩定化突變預測</h2>
      <p style="font-size:13px;color:var(--muted);margin-bottom:8px;">依據 Hazes & Dijkstra 幾何法則所篩選之去折疊態熵抑制（De Novo S-S）高潛力候選對：</p>
      <table>
        <thead>
          <tr>
            <th>殘基位點 1</th>
            <th>殘基位點 2</th>
            <th>Cα 距離</th>
            <th>Cβ 距離</th>
            <th>序列間距</th>
            <th>預估穩定化 ΔΔG</th>
            <th>預估 Tm 提升</th>
          </tr>
        </thead>
        <tbody>
          {ss_rows}
        </tbody>
      </table>
    </div>

    <!-- 區塊 3：逆轉譯與分子選殖引子 -->
    <div class="card">
      <h2>🧬 3. 宿主表現密碼子優化與 PCR 引子設計</h2>
      <table>
        <tr>
          <th>cDNA 全長</th>
          <td>{nt.get('cdna_len', 0)} bp</td>
          <th>整體 GC%</th>
          <td>{nt.get('gc_pct', 0.0):.1f}% (GC3%: {nt.get('gc3_pct', 0.0):.1f}%)</td>
        </tr>
        <tr>
          <th>密碼子適應指數 (CAI)</th>
          <td><span class="badge badge-success">{nt.get('cai', 0.0):.3f}</span></td>
          <th>5' 端前 30 nt 折疊 ΔG</th>
          <td><strong>{nt.get('dg_5prime', 0.0):.2f} kcal/mol</strong></td>
        </tr>
      </table>

      <h3 style="font-size:14px;margin-top:16px;color:var(--primary);">🎯 推薦 PCR 選殖引子 (5' → 3')：</h3>
      <div style="margin-top:8px;">
        <p style="font-size:12px;color:var(--muted);"><strong>5' Forward Primer:</strong> (退火 Tm = {primers.get('fwd_tm', 0.0):.1f} °C ｜ 長度 = {primers.get('fwd_len', 0)} nt)</p>
        <div class="seq-box">{primers.get('fwd_full', 'N/A')}</div>
      </div>
      <div style="margin-top:12px;">
        <p style="font-size:12px;color:var(--muted);"><strong>3' Reverse Primer:</strong> (退火 Tm = {primers.get('rev_tm', 0.0):.1f} °C ｜ 長度 = {primers.get('rev_len', 0)} nt)</p>
        <div class="seq-box">{primers.get('rev_full', 'N/A')}</div>
      </div>
    </div>

    <!-- 區塊 4：胺基酸與 cDNA 完整序列 -->
    <div class="card">
      <h2>📄 4. 蛋白質與 cDNA 序列明細</h2>
      <p style="font-size:12px;color:var(--muted);">胺基酸序列 FASTA:</p>
      <div class="seq-box" style="color:#A7F3D0;">>{title} | {l} aa&#10;{seq}</div>

      <p style="font-size:12px;color:var(--muted);margin-top:12px;">編碼區 cDNA 序列 (含終止密碼子):</p>
      <div class="seq-box">{nt.get('cdna_seq', 'N/A')}</div>
    </div>

    <div class="footer">
      <p>Generated by Protein Thermodynamics & Recombinant Design Platform ｜ Antigravity 2.0</p>
    </div>
  </div>
</body>
</html>"""
    return html
