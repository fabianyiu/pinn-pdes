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

Notes:
- Next time for quick testing I think no need exact equation... even just FDM with a forcing term could be accurate enough as reference? idk unless very complicated?
- Just wanted exact solution for comparision betweenn all. But in indusry FDM is the norm when uniformed then FEM is the next norm...

- This experiment is mostly as benchmarks. If you write a new piece of simulation software, you first run it on a "perfect cube" where the exact answer is known. If your code matches the exact solution to 10 decimal places, you know your simulation is trustworthy enough to handle the complex, real-world stuff.

- If you are in a classroom, you look for the exact solution. If you are designing a smartphone cooling system or a car engine, you run a Numerical Simulation

- Next experiment would be numerical simulation as benchmark.

- An alternative cleaner benchmark would use a sine IC u(x,0) = sin(πx) with homogeneous Dirichlet BCs u(0,t) = u(1,t) = 0, giving a single-mode exact solution u(x,t) = exp(-απ²t)sin(πx) with no IC/BC mismatch or Gibbs; however the current physical setup (instantaneous heat applied at one end) is retained to study how solvers handle the resulting discontinuity and Gibbs phenomenon.

- Used GO for quick visualisation 3D plot
