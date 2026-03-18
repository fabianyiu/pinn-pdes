# To run: streamlit run app.py

# Use Streamlit for rapid prototyping
import os
import sys
import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from heat_equation1D import heat_equation1D
from heat_equationFDM import heat_equation_FDM

# ── settings (match pinn_train.py) ───────────────────────────────────────────
alpha   = 1.0
x_left  = 0.0
x_right = 1.0
u_left  = 24.0
u_right = 100.0
u_ic    = 24.0
t_end   = 0.2
x_num   = 100
t_num   = 100
u_min   = 24.0
u_max   = 100.0

checkpoint_dir = os.path.join(os.path.dirname(__file__), "Results", "Checkpoints")

lambda_text = {
    "Neutral":   r"$\lambda_{pde}=1,\ \lambda_{ic}=1,\ \lambda_{bc}=1$",
    "BC heavy":  r"$\lambda_{pde}=1,\ \lambda_{ic}=1,\ \lambda_{bc}=10$",
    "IC heavy":  r"$\lambda_{pde}=1,\ \lambda_{ic}=10,\ \lambda_{bc}=1$",
    "PDE heavy": r"$\lambda_{pde}=10,\ \lambda_{ic}=1,\ \lambda_{bc}=1$",
}

lambda_values = {
    "Neutral":   (1, 1, 1),
    "BC heavy":  (1, 1, 10),
    "IC heavy":  (1, 10, 1),
    "PDE heavy": (10, 1, 1),
}

graph_title_map = {
    "Neutral": "Neutral Loss",
    "BC heavy": "BC dominated loss function",
    "IC heavy": "IC dominated loss function",
    "PDE heavy": "PDE dominated loss function",
}

runs = {
    "Neutral":   "neutral_pinn.pt",
    "BC heavy":  "bc_heavy_pinn.pt",
    "IC heavy":  "ic_heavy_pinn.pt",
    "PDE heavy": "pde_heavy_pinn.pt",
}

x_grid = np.linspace(x_left, x_right, x_num)
t_grid = np.linspace(0, t_end, t_num)


# ── cached solvers ────────────────────────────────────────────────────────────
@st.cache_data
def get_exact():
    solver = heat_equation1D(
        alpha=alpha, x_left=x_left, x_right=x_right,
        u_left=u_left, u_right=u_right, u_ic=u_ic,
    )
    return solver.exact(t=t_end, N=100, xNum=x_num, tNum=t_num)


@st.cache_data
def get_fdm():
    solver = heat_equation_FDM(
        alpha=alpha, x_left=x_left, x_right=x_right,
        u_left=u_left, u_right=u_right, u_ic=u_ic,
    )
    sol_raw, xlen, tlen, delt = solver.solve(t=t_end)
    x_fdm = np.linspace(x_left, x_right, xlen)
    t_fdm = np.linspace(0, delt * (tlen - 1), tlen)
    interp = RegularGridInterpolator(
        (t_fdm, x_fdm), sol_raw,
        method="linear", bounds_error=False, fill_value=None,
    )
    t_eval, x_eval = np.meshgrid(t_grid, x_grid, indexing="ij")
    pts = np.column_stack([t_eval.ravel(), x_eval.ravel()])
    return interp(pts).reshape(t_num, x_num)


@st.cache_data
def get_pinn(checkpoint_path):
    """Loads checkpoint, evaluates on shared grid, denormalises: u = pred * 76 + 24."""
    import torch.nn as nn

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    linear_keys = sorted(
        [k for k in state if k.endswith(".weight") and "net" in k],
        key=lambda k: int(k.split(".")[1]),
    )
    layers = []
    for key in linear_keys:
        out_f, in_f = state[key].shape
        layers.append(nn.Linear(in_f, out_f))
        if key != linear_keys[-1]:
            layers.append(nn.Tanh())

    class _pinn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)
        def forward(self, x, t):
            return self.net(torch.cat([x, t], dim=1))

    model = _pinn()
    model.load_state_dict(state)
    model.eval()

    t_eval, x_eval = np.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.tensor(x_eval.ravel(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(t_eval.ravel(), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_norm = model(x_flat, t_flat).numpy().ravel()

    return (u_norm * (u_max - u_min) + u_min).reshape(t_num, x_num)


def compute_metrics(u_pred, u_ref):
    mae  = float(np.mean(np.abs(u_pred - u_ref)))
    rmse = float(np.sqrt(np.mean((u_pred - u_ref) ** 2)))
    maxe = float(np.max(np.abs(u_pred - u_ref)))
    return mae, rmse, maxe


def surface_trace(sol, name, show_colorbar=False):
    z_min, z_max = float(np.min(sol)), float(np.max(sol))
    n_levels = 25
    step = (z_max - z_min) / (n_levels + 1) if z_max > z_min else 1.0
    return go.Surface(
        x=x_grid, y=t_grid, z=sol,
        name=name,
        showscale=show_colorbar,
        colorbar=dict(title="u (°C)") if show_colorbar else None,
        contours=dict(
            z=dict(
                show=True, start=z_min, end=z_max, size=step,
                usecolormap=False, color="black", highlightcolor="black",
                project=dict(z=False),
            ),
        ),
    )


def build_figure(u_exact, u_fdm, u_pinn, title):
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}] * 3],
        subplot_titles=["Exact (Fourier)", "FDM", "PINN"],
        horizontal_spacing=0.02,
    )
    camera = dict(
        eye=dict(x=-1.6, y=-1.6, z=1.2),  # tweak viewpoint as needed
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    )
    scene = dict(
        xaxis_title="x (space)",
        yaxis_title="t (time)",
        zaxis_title="u (°C)",
        zaxis=dict(range=[u_min, u_max]),
    )
    for col, (sol, name) in enumerate(
        [(u_exact, "Exact"), (u_fdm, "FDM"), (u_pinn, "PINN")], start=1
    ):
        show_colorbar = (col == 3)  # only on the last subplot (PINN)
        fig.add_trace(surface_trace(sol, name, show_colorbar=show_colorbar), row=1, col=col)
        fig.update_scenes(scene, row=1, col=col)
        fig.update_scenes(camera=camera, row=1, col=col)
    
    # Force Plotly to apply our default camera each render (avoid reusing prior UI state)
    fig.update_layout(
        uirevision=None,
        title_text=title,
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(size=26),
        margin=dict(t=60),
        height=430,
    )
    return fig


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Investigation on Physics Informed Neural Networks for Heat Equation", layout="wide")

st.markdown(
    """
    <style>
    /* tighten overall page padding/margins */
    .block-container { padding-top: 1rem; padding-left: 1.5rem; padding-right: 1.5rem; }

    /* reduce the vertical gap between the plot and the next section */
    div[data-testid="stPlotlyChart"] { margin-bottom: -0.5rem; }
    div[data-testid="stSubheader"] { margin-top: 0.25rem; margin-bottom: 0.25rem; }

    /* Hide the little link/anchor icon next to headings */
    a[aria-label="Link to this heading"],
    a.header-anchor,
    a.anchor-link,
    .stMarkdown a[href^="#"],
    .stMarkdown a[href*="#"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state for page navigation ────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "cover"

# ── cover page ────────────────────────────────────────────────────────────────
if st.session_state.page == "cover":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; font-size:2.8rem;'>Physics-Informed Neural Networks</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center; color:grey;'>When Physics Constraints Compete: PINN Behaviour and Trade-offs</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
<div style="font-size:1.25rem; line-height:1.5; max-width: 1200px; margin: 0 auto; padding: 0.4rem; text-align: center;">
  <p>
    <b>
      Physics-Informed Neural Networks don’t just solve equations, they negotiate between competing constraints.
    </b>
    By introducing a controlled mismatch between initial and boundary conditions, this project exposes a core issue: 
    <b>loss terms do not cooperate, they compete.</b> 
    We compare analytical solutions, finite difference methods, and PINNs to show how loss weighting determines what the model satisfies and what it ignores.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(r"\Large \mathcal{L}=\lambda_{pde}\mathcal{L}_{pde}+\lambda_{ic}\mathcal{L}_{ic}+\lambda_{bc}\mathcal{L}_{bc}.")

    st.markdown(
        """
        <div style="font-size:1.1rem; line-height:1.4; max-width: 850px; margin: 0 auto; padding: 0.15rem 1rem 0 1rem; text-align: center; color: #a8a8a8;">
          Here, <i>&lambda;<sub>pde</sub></i>, <i>&lambda;<sub>ic</sub></i>, and <i>&lambda;<sub>bc</sub></i> control how strongly the model enforces each objective.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="font-size:1.15rem; line-height:1.55; max-width: 950px; margin: 1.0rem auto; padding: 0.2rem 1rem; text-align: center; color: #d0d0d0;">
          The PINN is trained by minimising a weighted loss made up of three parts: the PDE residual, the initial condition, and the boundary condition.
          By changing the weights on these terms, we can see which constraints the model prioritizes and where error is pushed when they conflict.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("View Results", use_container_width=True):
            st.session_state.page = "results"
            st.rerun()

# ── results page ──────────────────────────────────────────────────────────────
else:
    st.title("When Physics Constraints Compete: PINN Behaviour and Trade-offs")
    st.markdown("<p style='font-size:1.4rem;'>Adjust constraint weighting to observe how the solution redistributes error across the domain.</p>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:1.0rem; font-weight:600; line-height:0.5; margin-top:0.0rem; margin-bottom:0.0rem;'>"
        "Loss weighting configuration:"
        "</div>",
        unsafe_allow_html=True,
    )
    run_label = st.selectbox("", list(runs.keys()))
    if run_label in lambda_values:
        lam_pde, lam_ic, lam_bc = lambda_values[run_label]
        st.markdown(
            rf"$$\Large \mathcal{{L}} = {lam_pde}\,\mathcal{{L}}_{{pde}} + {lam_ic}\,\mathcal{{L}}_{{ic}} + {lam_bc}\,\mathcal{{L}}_{{bc}}$$",
            unsafe_allow_html=True,
        )
    fname     = runs[run_label]
    ckpt      = os.path.join(checkpoint_dir, fname)

    if not os.path.exists(ckpt):
        st.error(f"Checkpoint not found: {fname}")
        st.stop()

    with st.spinner("Computing solutions ..."):
        u_exact = get_exact()
        u_fdm   = get_fdm()
        u_pinn  = get_pinn(ckpt)

    # 3D surface plot
    st.plotly_chart(
        build_figure(u_exact, u_fdm, u_pinn, graph_title_map.get(run_label, run_label)),
        use_container_width=True,
    )

    # Error metrics
    st.subheader("Error Metrics")
    mae_e, rmse_e, max_e = compute_metrics(u_pinn, u_exact)
    mae_f, rmse_f, max_f = compute_metrics(u_pinn, u_fdm)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**vs Exact (Fourier series)**")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (°C)",  f"{mae_e:.4f}")
        m2.metric("RMSE (°C)", f"{rmse_e:.4f}")
        m3.metric("Max (°C)",  f"{max_e:.4f}")

    with col2:
        st.markdown("**vs FDM**")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (°C)",  f"{mae_f:.4f}")
        m2.metric("RMSE (°C)", f"{rmse_f:.4f}")
        m3.metric("Max (°C)",  f"{max_f:.4f}")
