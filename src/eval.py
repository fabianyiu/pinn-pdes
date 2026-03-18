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

# Setup for heat equation 1D
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

# Path to checkpoints for reusability
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "Results", "Checkpoints")

# Runs for evaluation
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
    # Runs FDM on its own CFL-stable grid then interpolates onto the shared eval grid
    solver = heat_equation_FDM(
        alpha=alpha, x_left=x_left, x_right=x_right,
        u_left=u_left, u_right=u_right, u_ic=u_ic,
    )
    sol_raw, xlen, tlen, delt = solver.solve(t=t_end)
    # Create FDM grid for time and space, remember delt is timestep derived from delx and alpha
    x_fdm = np.linspace(x_left, x_right, xlen)
    t_fdm = np.linspace(0, delt * (tlen - 1), tlen)

    # Takes raw FDM solution irregular grid size because of CFL stability condition, resamples onto shared eval grid (100x100)
    # RegularGridInterpolator is a class that interpolates on a regular grid given the irregular grid points and values, it knows the grid points and values of the raw FDM solution.
    interp = RegularGridInterpolator(
        (t_fdm, x_fdm), sol_raw,
        method="linear", bounds_error=False, fill_value=None,
    )
    # Create target eval grid for time and space, grid is 100x100 for both time and space.
    t_eval, x_eval = np.meshgrid(t_grid, x_grid, indexing="ij")
    # Stack time and space eval grid points into a single array, ravel() flattens the grid into a 1D array. so 10000 points in 1D array paired i j in t_eval and x_eval.
    pts = np.column_stack([t_eval.ravel(), x_eval.ravel()])
    # Interpolate the raw FDM solution onto the shared eval grip (input of 10000 x 2 (time and space) points), then reshapes backt to 100x100.
    return interp(pts).reshape(t_num, x_num)


def build_pinn_from_state(state):
    # Infers network architecture from checkpoint weights and builds matching model.
    import torch.nn as nn
    # Collect all linear layers in the network, sorted by layer number to ensure correct order.
    linear_keys = sorted(
        [k for k in state if k.endswith(".weight") and "net" in k],
        key=lambda k: int(k.split(".")[1]),
    )
    # Build the layers of the network from the linear keys, each layer is a linear layer followed by a tanh activation function.
    layers = []
    for key in linear_keys:
        out_features, in_features = state[key].shape
        layers.append(nn.Linear(in_features, out_features))
        if key != linear_keys[-1]:
            layers.append(nn.Tanh())

    # No import from pinn.py to avoid model mismatch, reconstructing the exact architecutre from checkpoint weights, then loads no matter which model size produced at .pt.
    class _pinn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)
        def forward(self, x, t):
            return self.net(torch.cat([x, t], dim=1))

    return _pinn()


def get_pinn(checkpoint_path):
    # Loads checkpoint, evaluates on shared grid, denormalises: u = pred * 76 + 24.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_pinn_from_state(state)
    model.load_state_dict(state)
    model.eval()

    # Create target eval grid for time and space, grid is 100x100 for both time and space.
    t_eval, x_eval = np.meshgrid(t_grid, x_grid, indexing="ij")
    # Stack time and space eval grid points into a single array, ravel() flattens the grid into a 1D array. so 10000 points in 1D array paired i j in t_eval and x_eval.
    x_flat = torch.tensor(x_eval.ravel(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(t_eval.ravel(), dtype=torch.float32).unsqueeze(1)

    # No gradients needed for evaluation, saves memory and time.
    with torch.no_grad():
        # Forward pass call, input of 10000 x 2 (time and space) points, output of 10000 x 1 (temperature) points.
        u_norm = model(x_flat, t_flat).numpy().ravel()

    # Denormalise the predicted solution to the original scale, u = pred * 76 + 24, still in 10000 x 1 array.
    u_pred = u_norm * (u_max - u_min) + u_min
    # Reshape back to 100x100 for time and space.
    return u_pred.reshape(t_num, x_num)


def error_metrics(u_pred, u_ref, label="Exact"):
    # Calculate error metrics, mae is mean absolute error, rmse is root mean square error, maxe is maximum error.
    mae  = np.mean(np.abs(u_pred - u_ref))
    rmse = np.sqrt(np.mean((u_pred - u_ref) ** 2))
    maxe = np.max(np.abs(u_pred - u_ref))
    print(f"  vs {label:<6}  MAE={mae:.4f}  RMSE={rmse:.4f}  Max={maxe:.4f}")
    return mae, rmse, maxe


def surface_trace(sol, x, t, name):
    # Returns a Plotly Surface trace with contour lines in the same style as heat_equation1D.
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
    # 3-panel interactive 3D surface plot: Exact | FDM | PINN
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
    # Only once for each reference solution, no need to loop through runs as same for all runs to act as baseline ref
    u_exact = get_exact()
    u_fdm   = get_fdm()
    print(f"  Exact shape : {u_exact.shape}")
    print(f"  FDM shape   : {u_fdm.shape}")

    # Loop through each run, evaluate PINN on the shared eval grid, then denormalise to the original scale.
    for tag, fname, label in runs:
        ckpt = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(ckpt):
            print(f"\n[SKIP] {fname} not found at {ckpt}")
            continue

        print(f"\n── {label} ──")
        # Runs PINN on the shared eval grid, then denormalise to the original scale.
        u_pinn = get_pinn(ckpt)
        # Evaluate PINN on the shared eval grid, then denormalise to the original scale.
        error_metrics(u_pinn, u_exact, label="Exact")
        error_metrics(u_pinn, u_fdm,   label="FDM  ")

        # Plots the 3D surface plot: Exact | FDM | PINN for each test case.
        plot_3d(u_exact, u_fdm, u_pinn, label)


if __name__ == "__main__":
    main()
