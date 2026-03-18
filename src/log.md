- Dirichlet condition used 
- Instantaneous heat at the B.C not too realistic in that sense but simplest to implement for quick experiment. However, this makes exact solution much more convenient to calculate. For further studies FDM will be used as the norm...

## Exact:
- Exact solution had to be hand calculated first, B.C affects solution heavily. This process is hard to atutomated.
    - **Dirichlet**
    - Neumann
    - Robin
- Gibbs phenonenon is seen
    - oscillations awround BC where IC and BC don't agree
- Triple loop (tNum x xNum x N ),  slow and suitable for refrence only 


## FDM:
- delx = 1.0/99 used as 100 points on [0,1] so 99 gaps mimicing exact
- Not able to pick delt freely unlike exact solution
    - For exact solution you can choose how many time points to evaluate at where fomrula gives you the correct answer for t, no restriction.
    - FDM timestep delt is not free as it must satisfy CFL stabilty condition.
    - Courant number must be obeyed.
- no Gibbs phenomneon 

Keey in mind:
- Keep x:same (no interpolation needed)
- When comparing interpolaote function needed (this is in eval.py)

## PINN:
- IC, Boundary, Collocation points are all separate, as they are independently evaluated on loss func 
    - There are tutorials on YT on PINN that stack these but this is an alternative mini-batching strategy where the proportion of IC vs BC points vary randomly each epoch, it is kept separte here for explicit control over ewach loss term's contribution.
- No enabling of cuda as this is quick research project. But to implement only a few lines 
- PINN is hybrid supervised and unsupervised leanring. Supervised with labels at IC, BC but unsupervised and finding pattern for collocation points BC

4 runs:
    1. Neutral: lam_pde=1, lam_ic=1, lam_bc=1 (Done labeled in results neutral_PINN)
    2. BC heavy: lam_bc=10, lam_pde=1, lam_ic=1
    3. IC heavy: lam_ic=10, lam_pde=1, lam_bc=1
    4. PDE heavy: lam_pde=10, lam_ic=1, lam_bc=1

No early stopping implemented but rather fixed epoch to observe where each configuration ends up.

To change run need to switch out:
- Lambda for loss value wanted to change (Line 60)
- Residual title (Line 120)
- Plot saved name (Line 124)
- Model checkpoint saving file name (Line 129)

Expected observations:
BC heavy → BCs satisfied quickly, but PDE physics may be wrong in the interior.
IC heavy → initial condition fits well, but solution may drift over time.
PDE heavy → physics satisfied in interior, but may violate BCs/IC.

Residual loss plot implemented like standard CFD software
    - Can clearly see behaviour refer to first try neutral_pinn_lossV1 sys.png BC, behaviour
    - Files are saved to Results/Plots for reference.
Problem encoutnered:
    - On first try residual loss for BC wouldn't drop below e1
    Tried:
    - Expanding out architecture to larger neural net and one extra layer
    - Reducing alpha to a more realistic 0.0001 
    - Altered learning rate to lower number.
        - Or use of scehduler (rule that changes the learing rate during training)
    - Try lam_BC =  higher and see if this helps (which acutally goes against the study about changing lambda for lr)
Solution:
    - Found out that scailing was the issue 
    - IC and U were in similar ranges but BC always had a soft constraint training of 24 and 100. 
    - When this was linearly scaled for training then rescaled for evaluation. Problem was eliminated. Eval reverese by doing u_actual = u_normalised * 76 + 24
    - Normalise to 0-1 range for training
    - Training results straight away down to loss approx e-3
    - No need for standardScaler as it fits mean/std on unknown data; our u range is fixed by BCs (24–100), so can just rescale by hand.
    - No need and can't sacle for PDE part since unsupervised but with x and t in nice range it hsould be fine. Just minimising r^2

    Remember to close interactive residual plot to save the .pt file

## app.py 
- The app was built for rapid iteration, focusing on testing PINN behaviour rather than UI polish
- Core physics, solvers, and visualisation code were developed independently; the interface acts as a lightweight interaction layer
- Plotting components were reused from earlier experiments to accelerate development

## Notes and thoughts:
Notes:
- For quick experiments, an exact solution probably isn’t always needed. A well-resolved FDM solution (maybe with a forcing term) is usually good enough as a reference, especially once things get more complex and exact solutions aren’t available anyway.
- Used the exact solution here mainly for clean comparison across all methods. In practice though, you’d almost always rely on numerical methods — FDM for simpler structured setups, FEM once geometry gets more complicated.
- This is basically a benchmarking setup. Same idea as testing a solver on a simple, known case first before trusting it on real problems.
- Exact solutions are more of a classroom / controlled setting thing. In real applications (cooling systems, engines, etc.), everything is numerical because the physics + geometry are too messy.
- Next step: switch to numerical solutions as the main benchmark instead of exact.
- A cleaner version of this experiment would use something like 
  u(x,0) = sin(πx) with u(0,t) = u(1,t) = 0 → gives a single-mode solution with no IC/BC mismatch and no Gibbs effects.
- But I kept the current setup (instant heat at the boundary) on purpose, the mismatch creates discontinuities, which is useful for seeing how each method handles it.
- Used Plotly (graph_objects) for quick 3D visualisation.
