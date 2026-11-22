import numpy as np
import plotly.graph_objects as go


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
def create_energy_figure(model_a, model_b):
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
                name=f"{name} States",
                marker=dict(size=9),
                showlegend=False
            )
        )

    fig.update_layout(
        title="Protein Folding Free Energy Landscape",
        xaxis_title="Reaction Coordinate",
        yaxis_title="Gibbs Free Energy (kJ/mol)",
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend_title="Models"
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
    height=720
):
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
        "smooth": "Smooth Folding Funnel",
        "rugged": "Rugged Energy Landscape",
        "multi": "Multi-basin Energy Landscape"
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

    zaxis_cfg = dict(title="Effective Energy (kJ/mol)")
    if z_range is not None:
        zaxis_cfg["range"] = z_range

    fig.update_layout(
        title=chart_title,
        scene=dict(
            xaxis_title="Coord 1",
            yaxis_title="Coord 2",
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


# =====================================
# 3. 3D C-alpha Backbone Model
# =====================================
def create_ca_backbone_figure(
    chain_results,
    selected_model=0,
    color_mode="chain",
    show_markers=True,
    height=650
):
    fig = go.Figure()

    target_chains = [item for item in chain_results if item["model"] == selected_model]

    for item in target_chains:
        coords = np.array(item["ca_coords"])
        if len(coords) == 0:
            continue

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        labels = item["res_labels"]
        chain_id = item["chain"]
        n_res = len(coords)

        mode = "lines+markers" if show_markers else "lines"

        if color_mode == "sequence":
            seq_idx = np.arange(1, n_res + 1)
            line_cfg = dict(
                width=5,
                color=seq_idx,
                colorscale="Turbo",
                colorbar=dict(title="Residue Index<br>(N → C)") if len(target_chains) == 1 else None
            )
            marker_cfg = dict(
                size=3.5,
                color=seq_idx,
                colorscale="Turbo"
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
                name=f"Chain {chain_id} ({n_res} aa)",
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
                marker=dict(size=6, symbol="diamond"),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title=f"3D Cα Backbone Trace (Model {selected_model})",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data"
        ),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(title="Polypeptide Chains")
    )

    return fig

# =====================================
# 4. Residue-Residue Contact Map
# =====================================
def create_contact_map_figure(
    chain_item,
    cutoff=8.0,
    map_mode="binary",
    height=620
):
    coords = np.array(chain_item["ca_coords"])
    chain_id = chain_item["chain"]
    model_id = chain_item["model"]
    seq = chain_item["sequence"]
    n_res = len(coords)

    if n_res == 0:
        return go.Figure()

    # 計算 N x N Cα 歐氏距離矩陣 (Å)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    res_indices = np.arange(1, n_res + 1)
    hover_text = [
        [
            f"Res {i+1} ({seq[i] if i < len(seq) else '?'}) ↔ "
            f"Res {j+1} ({seq[j] if j < len(seq) else '?'})<br>"
            f"Cα Distance: {dist_matrix[i, j]:.2f} Å"
            for j in range(n_res)
        ]
        for i in range(n_res)
    ]

    if map_mode == "binary":
        # 二元接觸矩陣：距離小於閾值為 1，其餘為 0
        z_data = (dist_matrix <= cutoff).astype(int)
        colorscale = [[0.0, "#F8FAFC"], [1.0, "#1E3A8A"]]
        colorbar_cfg = dict(
            title="Contact",
            tickvals=[0, 1],
            ticktext=[f"> {cutoff:.1f} Å", f"≤ {cutoff:.1f} Å"]
        )
        title_text = f"Cα Contact Map — Model {model_id} Chain {chain_id} (Cutoff ≤ {cutoff:.1f} Å)"
    else:
        # 連續距離熱圖
        z_data = dist_matrix
        colorscale = "Viridis_r"
        colorbar_cfg = dict(title="Distance (Å)")
        title_text = f"Cα Distance Matrix — Model {model_id} Chain {chain_id}"

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_data,
                x=res_indices,
                y=res_indices,
                text=hover_text,
                hoverinfo="text",
                colorscale=colorscale,
                colorbar=colorbar_cfg,
                showscale=True
            )
        ]
    )

    fig.update_layout(
        title=title_text,
        xaxis=dict(title="Residue Index", constrain="domain"),
        yaxis=dict(title="Residue Index", scaleanchor="x", scaleratio=1),
        height=height,
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig

# =====================================
# 5. Ramachandran Plot (Phi / Psi)
# =====================================
def create_ramachandran_figure(chain_item, height=620):
    rama_data = chain_item.get("rama_data", [])
    chain_id = chain_item["chain"]
    model_id = chain_item["model"]

    fig = go.Figure()

    if not rama_data:
        fig.update_layout(
            title=f"Ramachandran Plot — Model {model_id} Chain {chain_id} (No continuous backbone angles)",
            height=height
        )
        return fig

    # 1. 繪製經典二級結構允許區背景框
    # Beta-sheet 區域 (左上)
    fig.add_shape(
        type="rect", x0=-165, y0=90, x1=-45, y1=175,
        fillcolor="rgba(59, 130, 246, 0.14)",
        line=dict(color="rgba(59, 130, 246, 0.45)", width=1, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=-105, y=135, text="β-Sheet Region",
        showarrow=False, font=dict(size=11, color="#1D4ED8")
    )

    # Right-handed alpha-helix 區域 (左中下)
    fig.add_shape(
        type="rect", x0=-110, y0=-70, x1=-35, y1=-15,
        fillcolor="rgba(16, 185, 129, 0.16)",
        line=dict(color="rgba(16, 185, 129, 0.55)", width=1, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=-72, y=-42, text="Right-handed α-Helix",
        showarrow=False, font=dict(size=11, color="#047857")
    )

    # Left-handed alpha-helix 區域 (右上)
    fig.add_shape(
        type="rect", x0=30, y0=10, x1=90, y1=75,
        fillcolor="rgba(245, 158, 11, 0.15)",
        line=dict(color="rgba(245, 158, 11, 0.50)", width=1, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=60, y=42, text="Left-handed α",
        showarrow=False, font=dict(size=10, color="#B45309")
    )

    # 座標軸零點十字基準線
    fig.add_hline(y=0, line_width=1, line_color="#CBD5E1")
    fig.add_vline(x=0, line_width=1, line_color="#CBD5E1")

    # 2. 依殘基類型分組 (一般胺基酸、Glycine、Proline)
    groups = [
        ("General Residues", [d for d in rama_data if d["res_name"] not in ("GLY", "PRO")], "circle", "#2563EB"),
        ("Glycine (GLY)", [d for d in rama_data if d["res_name"] == "GLY"], "diamond", "#DC2626"),
        ("Proline (PRO)", [d for d in rama_data if d["res_name"] == "PRO"], "square", "#7C3AED"),
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
        title=f"Ramachandran Plot — Model {model_id} Chain {chain_id}",
        xaxis=dict(
            title="φ (Phi) Angle (degrees)",
            range=[-180, 180],
            dtick=60,
            constrain="domain"
        ),
        yaxis=dict(
            title="ψ (Psi) Angle (degrees)",
            range=[-180, 180],
            dtick=60,
            scaleanchor="x",
            scaleratio=1
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(title="Residue Type")
    )

    return fig

from plotly.subplots import make_subplots


# =====================================
# 6. Arrhenius / Eyring Temperature Sweep
# =====================================
def create_kinetics_sweep_figure(sweep_a, sweep_b, height=560):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Rate Constants vs. Temperature (log₁₀ k)",
            "Arrhenius Plot (ln k vs. 1000/T)"
        ),
        horizontal_spacing=0.10
    )

    # --- 左圖：log10(k) vs Temperature (K) ---
    # Model A (藍色系)
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

    # Model B (橘色系)
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

    # 標註熔解溫度 Tm 交點
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

    # --- 右圖：Arrhenius Plot (ln k vs 1000/T) ---
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

    fig.update_xaxes(title_text="Temperature T (K)", row=1, col=1)
    fig.update_yaxes(title_text="log₁₀ k (s⁻¹)", row=1, col=1)
    fig.update_xaxes(title_text="1000 / T (K⁻¹)", row=1, col=2)
    fig.update_yaxes(title_text="ln k", row=1, col=2)

    fig.update_layout(
        title="Arrhenius / Eyring Temperature Sweep Kinetics",
        template="plotly_white",
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5)
    )

    return fig