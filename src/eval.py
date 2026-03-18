# Evaluate PINN on grid, compare with exact and FDM
import os
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(__file__))
from pinn import pinn
from heat_equation1D import heat_equation1D
from heat_equationFDM import heat_equation_FDM

# Settings
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

checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "Results", "Checkpoints")

runs = [
    ("neutral",   "neutral_pinn.pt",   "Neutral  (lam_pde=1, lam_ic=1, lam_bc=1)"),
    ("bc_heavy",  "bc_heavy_pinn.pt",  "BC heavy (lam_bc=10)"),
    ("ic_heavy",  "ic_heavy_pinn.pt",  "IC heavy (lam_ic=10)"),
    ("pde_heavy", "pde_heavy_pinn.pt", "PDE heavy (lam_pde=10)"),
]

# Shared evaluation grid
x_grid = np.linspace(x_left, x_right, x_num)
t_grid = np.linspace(0, t_end, t_num)


def get_exact():
    solver = heat_equation1D(
        alpha=alpha, x_left=x_left, x_right=x_right,
        u_left=u_left, u_right=u_right, u_ic=u_ic,
    )
    return solver.exact(t=t_end, N=100, xNum=x_num, tNum=t_num)


def get_fdm():
    """Runs FDM on its own CFL-stable grid then interpolates onto the shared eval grid."""
    solver = heat_equation_FDM(
        alpha=alpha, x_left=x_left, x_right=x_right,
        u_left=u_left, u_right=u_right, u_ic=u_ic,
    )
    sol_raw, xlen, tlen = solver.solve(t=t_end)
    x_fdm = np.linspace(x_left, x_right, xlen)
    delx  = (x_right - x_left) / (xlen - 1)
    delt  = 0.4 * delx**2 / alpha
    t_fdm = np.linspace(0, delt * (tlen - 1), tlen)

    interp = RegularGridInterpolator(
        (t_fdm, x_fdm), sol_raw,
        method="linear", bounds_error=False, fill_value=None,
    )
    t_eval, x_eval = np.meshgrid(t_grid, x_grid, indexing="ij")
    pts = np.column_stack([t_eval.ravel(), x_eval.ravel()])
    return interp(pts).reshape(t_num, x_num)


def build_pinn_from_state(state):
    """Infers network architecture from checkpoint weights and builds matching model."""
    import torch.nn as nn
    linear_keys = sorted(
        [k for k in state if k.endswith(".weight") and "net" in k],
        key=lambda k: int(k.split(".")[1]),
    )
    layers = []
    for key in linear_keys:
        out_features, in_features = state[key].shape
        layers.append(nn.Linear(in_features, out_features))
        if key != linear_keys[-1]:
            layers.append(nn.Tanh())

    class _pinn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)
        def forward(self, x, t):
            return self.net(torch.cat([x, t], dim=1))

    return _pinn()


def get_pinn(checkpoint_path):
    """Loads checkpoint, evaluates on shared grid, denormalises: u = pred * 76 + 24."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_pinn_from_state(state)
    model.load_state_dict(state)
    model.eval()

    t_eval, x_eval = np.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.tensor(x_eval.ravel(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(t_eval.ravel(), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_norm = model(x_flat, t_flat).numpy().ravel()

    u_pred = u_norm * (u_max - u_min) + u_min
    return u_pred.reshape(t_num, x_num)


def error_metrics(u_pred, u_ref, label="Exact"):
    mae  = np.mean(np.abs(u_pred - u_ref))
    rmse = np.sqrt(np.mean((u_pred - u_ref) ** 2))
    maxe = np.max(np.abs(u_pred - u_ref))
    print(f"  vs {label:<6}  MAE={mae:.4f}  RMSE={rmse:.4f}  Max={maxe:.4f}")
    return mae, rmse, maxe


def surface_trace(sol, x, t, name):
    """Returns a Plotly Surface trace with contour lines in the same style as heat_equation1D."""
    z_min, z_max = float(np.min(sol)), float(np.max(sol))
    n_levels = 25
    step = (z_max - z_min) / (n_levels + 1) if z_max > z_min else 1.0
    return go.Surface(
        x=x, y=t, z=sol,
        name=name,
        showscale=False,
        contours=dict(
            z=dict(
                show=True,
                start=z_min,
                end=z_max,
                size=step,
                usecolormap=False,
                color="black",
                highlightcolor="black",
                project=dict(z=False),
            ),
        ),
    )


def plot_3d(u_exact, u_fdm, u_pinn, run_label):
    """3-panel interactive 3D surface plot: Exact | FDM | PINN."""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["Exact (Fourier)", "FDM", "PINN"],
        horizontal_spacing=0.02,
    )

    scene = dict(
        xaxis_title="x (space)",
        yaxis_title="t (time)",
        zaxis_title="u (temp °C)",
        zaxis=dict(range=[u_min, u_max]),
    )

    for col, (sol, name) in enumerate(
        [(u_exact, "Exact"), (u_fdm, "FDM"), (u_pinn, "PINN")], start=1
    ):
        fig.add_trace(surface_trace(sol, x_grid, t_grid, name), row=1, col=col)
        fig.update_scenes(scene, row=1, col=col)

    fig.update_layout(
        title_text=run_label,
        title_x=0.5,
        height=550,
        width=1400,
    )

    fig.show()


def main():
    print("Computing reference solutions ...")
    u_exact = get_exact()
    u_fdm   = get_fdm()
    print(f"  Exact shape : {u_exact.shape}")
    print(f"  FDM shape   : {u_fdm.shape}")

    for tag, fname, label in runs:
        ckpt = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(ckpt):
            print(f"\n[SKIP] {fname} not found at {ckpt}")
            continue

        print(f"\n── {label} ──")
        u_pinn = get_pinn(ckpt)

        error_metrics(u_pinn, u_exact, label="Exact")
        error_metrics(u_pinn, u_fdm,   label="FDM  ")

        plot_3d(u_exact, u_fdm, u_pinn, label)


if __name__ == "__main__":
    main()
