# Train PINN, define loss term and training loop, saves checkpoints
import os
import torch
from pinn import pinn
import random
import numpy as np
import matplotlib.pyplot as plt


# Settings
N = 200 # Grid settings
t_end = 0.2
# Normalise u_ic, u_left, and u_right for trianing so that the network doesn't have to learn the offset and scale, in eval reverese by doing u_actual = u_normalised * 76 + 24
u_ic = 24.0
u_ic  = (u_ic -24)/76 # Normalise to 0-1 range for training
u_left = 24.0
u_left = (u_left - 24)/76 # Normalise to 0-1 range for training
u_right = 100.0
u_right = (u_right - 24)/76 # Normalise to 0-1 range for training
alpha = 1
n_epochs = 62500

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# For BC: (0, t_random, 24), (1, t_random, 100) — x is fixed, t varies.
# For IC: (x_random, 0, 24) — t is fixed, x varies.
# For PDE: (x_random, t_random, ?) — no u value needed, just the location.

# IC: t=0, x random in [0,1]
x_ic = torch.rand(N, 1)          # 200 random x values (in space as time t = 0 constrained)
t_ic = torch.zeros(N, 1)         # all t=0
u_ic_vals = torch.full((N, 1), u_ic)
# BC left: x=0, t random
x_bc_left = torch.zeros(N, 1)    # all x=0
t_bc_left = torch.rand(N, 1) * t_end # 200 random values (in time that the BC of u_left is constrained)
u_bc_left_vals = torch.full((N, 1), u_left)
# BC right: x=1, t random
x_bc_right = torch.ones(N, 1)    # all x=1
t_bc_right = torch.rand(N, 1) * t_end # 200 random values (in time that the BC of u_right is constrained)
u_bc_right_vals = torch.full((N, 1), u_right)


# PDE collocation: random interior (no u label)
x_pde = torch.rand(2000, 1, requires_grad=True)
t_pde = torch.rand(2000, 1, requires_grad=True) * t_end

def pde_residual(model, x, t, alpha):
    u = model(x, t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    return u_t - alpha * u_xx   # should be 0


# PINN specific: MSE loss funciton to compute losses for each PDE, IC, BC where the ratios can be changed.
def compute_loss(model, x_pde, t_pde, lam_bc=1.0, lam_ic=1.0, lam_pde=1.0):
    # IC loss
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_vals)**2)

    # BC loss
    loss_bc = torch.mean((model(x_bc_left, t_bc_left) - u_bc_left_vals)**2) \
            + torch.mean((model(x_bc_right, t_bc_right) - u_bc_right_vals)**2)

    # PDE loss
    r = pde_residual(model, x_pde, t_pde, alpha)
    loss_pde = torch.mean(r**2)

    return lam_pde * loss_pde + lam_ic * loss_ic + lam_bc * loss_bc, loss_ic, loss_bc, loss_pde

def main():
    model = pinn()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Toggle train mode for good rpactice 
    model.train()
    
    # Live potting
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    epochs_log, total_log, pde_log, ic_log, bc_log = [], [], [], [], []
    
    for epoch in range(n_epochs):
        optimiser.zero_grad() # Care this step easy to forget and make mistake, clears gradients

        # PINN specific, need to resampling these points every epoch for better converage
        # As not fixed data points with u value, no labels exist so enforcing an equation, not fitting data.
        x_pde = torch.rand(2000, 1, requires_grad=True)
        t_pde = torch.rand(2000, 1, requires_grad=True) * t_end

        loss, loss_ic, loss_bc, loss_pde = compute_loss(model, x_pde, t_pde)
        loss.backward()
        optimiser.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4e}")
            print(f"IC Loss {epoch}: loss = {loss_ic}")
            print(f"BC Loss {epoch}: loss = {loss_bc}")
            print(f"PDE Loss {epoch}: loss = {loss_pde}")
            print("==================================================")

        # Implement live monitor of these losses like a convergence graph monitor in starCCM
        if epoch % 50 == 0:
            epochs_log.append(epoch)
            total_log.append(loss.item())
            pde_log.append(loss_pde.item())
            ic_log.append(loss_ic.item())
            bc_log.append(loss_bc.item())
            ax.cla()  # clear axes
            ax.plot(epochs_log, total_log, label="Total")
            ax.plot(epochs_log, pde_log,   label="PDE")
            ax.plot(epochs_log, ic_log,    label="IC")
            ax.plot(epochs_log, bc_log,    label="BC")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")  # log scale so you can see all curves
            ax.legend()
            ax.set_title(f"Neutral PINN Loss")
            plt.pause(0.01)

    plt.ioff()
    plt.savefig("results/Plots/neutral_pinn_loss.png")
    plt.show()

    os.makedirs("checkpoints", exist_ok=True)
    # Care when changing the compute loss manually chagne the name of saved pt file as well... 
    torch.save(model.state_dict(), "Results/Checkpoints/neutral_pinn.pt")

if __name__ == "__main__":
    main()
