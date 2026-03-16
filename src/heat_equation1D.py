# Class for generating exact solution for heat equation
# Domain input 
# B.C (Dirichlet)
# I.C (constant: rod starts at uniform u_ic)
# Simulating 1D rod at room temp with Dirichlet BC: one end at room temp, other end heated

import numpy as np
import plotly.graph_objects as go

class heat_equation1D:
    def __init__(self, alpha=1.0, x_left=0.0, x_right=1.0, u_left=24.0, u_right=100.0, u_ic=24.0):
        self.alpha = alpha
        self.x_left = x_left
        self.x_right = x_right
        # Dirichlet BCs (constant temperatures at each end, enforced for all t)
        self.u_left = u_left   # room temp end
        self.u_right = u_right # heated end (heat applied/source, e.g. bunsen burner)
        self.u_ic = u_ic       # initial temp (constant 24°C) for all x at t = 0

    # t      : end time (s); solution is computed from 0 to t.
    # N      : number of Fourier modes; higher N = better approximation to the infinite series.
    # xNum   : number of spatial grid points along the rod.
    # tNum   : number of time points in [0, t].
    # Returns: 2D array of shape (tNum, xNum); sol[i,j] = u(x_grid[j], t_grid[i]).
    # Note exact solution had to be re-arranged by hand first through B.C and I.C. 
    # Key assumptions:  Dirchlet B.C. and u_left = u_initial
    def exact(self,t=1,N=100, xNum =100, tNum =100):
        
        # Declaration
        L = self.x_right - self.x_left # Length of rod
        x_grid = np.linspace(self.x_left,self.x_right,xNum)
        t_grid = np.linspace(0,t,tNum)
        sol = []

        # Find u_steady(x)
        # For 1D heat equation we know that the uxx = 0 in steady state, so the function must be a straight line 
        # so for x = 0, u = u1 and x = L,  u = u2 from u(x) = mx + c 
        # exact = u_steady + u_transient
        for t in t_grid:
            for x in x_grid:
                u_steady = (self.u_right - self.u_left) / L * (x - self.x_left) + self.u_left
                u_transient = 0.0
                for n in range(1, N + 1):
                    k_n = n * np.pi / L
                    # Improved more general form so u_intiial doesn't have to be u_left but will cancel out if it is
                    b_n = (2 / (n * np.pi)) * ((self.u_ic - self.u_left) * (1 - (-1)**n) + (self.u_right - self.u_left) * (-1)**n)
                    # This will only work if u initial =  u left 
                    #b_n = 2 * (self.u_right - self.u_left) / (n * np.pi) * ((-1) ** n)
                    # This is general formula with incorporated Bn 
                    u_transient += b_n * np.sin(k_n * (x - self.x_left)) * np.exp(-self.alpha * k_n**2 * t)  
                # Store in array
                sol.append(u_steady + u_transient) # note sol will just be a super long array [u(x0,t0), u(x1,t0), ..., u(x99,t0), u(x0,t1), ..., u(x99,t99)] need re-shape to 100 * 100
        return np.array(sol).reshape(tNum, xNum) # reshape is row x column 


    def visual(self,sol, x_grid, t_grid):
        X, T = np.meshgrid(x_grid, t_grid)
        z_min, z_max = float(np.min(sol)), float(np.max(sol))
        n_levels = 25
        step = (z_max - z_min) / (n_levels + 1) if z_max > z_min else 1.0
        fig = go.Figure(data=[go.Surface(
                x=x_grid, y=t_grid, z=sol,
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
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title='x (space)',
                yaxis_title='t (time)',
                zaxis_title='u (temperature)',
            ),
            title='Heat equation solution'
        )
        fig.show()  # opens in browser or renders in Jupyter

#Testing
pde = heat_equation1D()
t_end, xNum, tNum = 0.1, 100, 100
sol = pde.exact(t=t_end, xNum=xNum, tNum=tNum)
x_grid = np.linspace(pde.x_left, pde.x_right, xNum)
t_grid = np.linspace(0, t_end, tNum)
pde.visual(sol, x_grid, t_grid)
