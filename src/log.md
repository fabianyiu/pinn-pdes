- Dirichlet condition used 
- Instantaneous heat at the B.C not too realistic in that sense but simplest to implement for quick experiment. However, this makes exact solution much more convenient to calculate. For further studies FDM will be used as the norm...

Solution convention:
Sol[i,j] Time and Spatial convention (row = time, column = space).

Exact:
- Exact solution had to be hand calculated first, B.C affects solution heavily. This process is hard to atutomated.
    - **Dirichlet**
    - Neumann
    - Robin
- Gibbs phenonenon is seen
    - oscillations awround BC where IC and BC don't agree
- Triple loop (tNum x xNum x N ),  slow and suitable for refrence only 


FDM:
- delx = 1.0/99 used as 100 points on [0,1] so 99 gaps mimicing exact
- cannot pick delt freely unlike exact solution
    - for exact solution you can choose how many time points to evaluate at where fomrula gives you the correct answer for t, no restriction.
    - FDM timestep delt is not free as it must satisfy stabilty condition 
    - Courant number must be obeyed.
- no Gibbs
- FDM has different t_grid (more steps because of stability) but same x_grid

Problem:
Keep x:same (no interpolation needed)
Interpolaote t: FDM at the exact t_grid values 
Then find diff and compare.

PINN:
- IC, Boundary, Collocation points are all separate, as they are independently evaluated on loss func 
    - There are tutorials on YT on PINN that stack these but this is an alternative mini-batching strategy where the proportion of IC vs BC points vary randomly each epoch, it is kept separte here for explicit control over ewach loss term's contribution.
- No enabling of cuda as this is quick research project. But to implement only a few lines 
- PINN hybrid supervised and unsupervised leanring. Supervised with labels at IC, BC but unsupervised and finding pattern for collocation points BC

4 runs:
1. Neutral: lam_pde=1, lam_ic=1, lam_bc=1 (Done labeled in results neutral_PINN)
2. BC heavy: lam_bc=10, lam_pde=1, lam_ic=1
3. IC heavy: lam_ic=10, lam_pde=1, lam_bc=1
4. PDE heavy: lam_pde=10, lam_ic=1, lam_bc=1

No early stopping implemented but rather fixed epoch to observe where each end up.

To change run need to switch out:
- Lambda for loss value wanted to change (Line 60)
- Residual title (Line 120)
- Plot saved name (Line 124)
- Model checkpoint saving file name (Line 129)

Expected observations:
BC heavy → BCs satisfied quickly, but PDE physics may be wrong in the interior.
IC heavy → initial condition fits well, but solution may drift over time.
PDE heavy → physics satisfied in interior, but may violate BCs/IC.


- loss plot implemented like starccm
    - Can clearly see behaviour refer to first try neutral_pinn_lossV1 sys.png BC, behaviour 
    Factors/ideas that can improve this and which were tested 
    - Expanding out architecture? why? or shrinking idk
    - Alpha is the problem? Thermal diffusivity for most material much smaller? 0.0001 for steel? larger alpha makes u_xxx term harder to learn 
    - Learning rate too high for late learning 
        - maybe reasonable to add a sheduler for late training 
    - Try lam_BC =  higher and see if this helps 
    - Another thing is that the loss converfged at high value 10^1 stuck in local minimum or doesn't have enough cpaicity, known failuremode in loss landscape get stuck 

Tried to do normalisation of IC, BCs 
    - Normalise to 0-1 range for training
    - Eval reverese by doing u_actual = u_normalised * 76 + 24
        - Training results straight away down to loss approx e-3
        - No need for standardScaler as it fits mean/std on unknown data; our u range is fixed by BCs (24–100), so can just rescale by hand.
        - No need and can't sacle for PDE part since unsupervised but with x and t in nice range it hsould be fine. Just minimising r^2

- No early stopping implemented beacuse want to keep consistent on all four setups. Fixed epoc to observe where each setup ends up

Remember to close interactive residual plot to save the .pt file

Notes:
- Next time for quick testing I think no need exact equation... even just FDM with a forcing term could be accurate enough as reference? idk unless very complicated?
- Just wanted exact solution for comparision betweenn all. But in indusry FDM is the norm when uniformed then FEM is the next norm...

- This experiment is mostly as benchmarks. If you write a new piece of simulation software, you first run it on a "perfect cube" where the exact answer is known. If your code matches the exact solution to 10 decimal places, you know your simulation is trustworthy enough to handle the complex, real-world stuff.

- If you are in a classroom, you look for the exact solution. If you are designing a smartphone cooling system or a car engine, you run a Numerical Simulation

- Next experiment would be numerical simulation as benchmark.

- An alternative cleaner benchmark would use a sine IC u(x,0) = sin(πx) with homogeneous Dirichlet BCs u(0,t) = u(1,t) = 0, giving a single-mode exact solution u(x,t) = exp(-απ²t)sin(πx) with no IC/BC mismatch or Gibbs; however the current physical setup (instantaneous heat applied at one end) is retained to study how solvers handle the resulting discontinuity and Gibbs phenomenon.

- Used GO for quick visualisation 3D plot
