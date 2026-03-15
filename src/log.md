- Dirichlet condition used 
- Insetantaneous heat at the B.C not too realistic in that sense but simplest to implement for quick experiment 


Exact:
- Exact solution had to be hand calculated first, B.C affects solution heavily.
    - **Dirichlet**
    - Neumann
    - Robin
- Gibbs phenonenon is seen
    - oscillations awround BC where IC and BC don't agree 

FDM:
- delx = 1.0/99 used as 100 points on [0,1] so 99 gaps mimicing exact
- cannot pick delt freely unlike exact solution
    - for exact solution you can choose how many time points to evaluate at where fomrula gives you the correct answer for t, no restriction.
    - FDM timestep delt is not free as it must satisfy stabilty condition 
    - Courant number must be obeyed.
- no Gibbs

PINN:

