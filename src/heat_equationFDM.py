#Numerical methods class

import numpy as np
import plotly.graph_objects as go

class heat_equation_FDM:
    def __init__(self, alpha=1.0, x_left=0.0, x_right=1.0, u_left=24.0, u_right=50.0, u_ic=24.0):
        self.alpha = alpha
        self.x_left = x_left
        self.x_right = x_right
        # Dirichlet BCs (constant temperatures at each end, enforced for all t)
        self.u_left = u_left   # room temp end
        self.u_right = u_right # heated end (heat applied/source, e.g. bunsen burner)
        self.u_ic = u_ic       # initial temp (constant 24°C) for all x at t = 0
        self.delx = 1.0/99    # xNum based from exact

    def solve(self,t=0.1):
        # Decalration 
        xlen = int((self.x_right-self.x_left)/self.delx) +1  # Number of intervals + 1
        delt = 0.4 * self.delx**2 / self.alpha               # r = 0.4 < 0.5 always stable
        tlen = int(t/delt)                                   # Number of intervals + 1
        sol = np.zeros((tlen,xlen))
        # Rememebr row x column is the convention
        sol[0,:] = self.u_ic                                 # Set IC (+1 row = +1 timestep)
        sol[:,0] = self.u_left                               # Set BC (left column)
        sol[:,-1] = self.u_right                             # Set BC (right column)
        # Courant number 
        r = self.alpha * delt / self.delx**2
        # Pre-define index 
        # Loop from IC to end -1 because of python 0 index
        for i in range(0,tlen-1):
            # Spatial loop 
            for j in range(1,xlen-1):
                sol[i+1, j] = sol[i, j] + r * (sol[i, j+1] - 2*sol[i, j] + sol[i, j-1])
        
        return sol, xlen, tlen



    def visual(self, sol, x_grid,t_grid):
        z_min, z_max = float(self.u_left), float(np.max(sol))
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
            title='Heat equation solution (FDM)'
        )
        fig.show()

#Testing 
pde = heat_equation_FDM()
sol, xlen, tlen = pde.solve(t=0.1)
x_grid = np.linspace(pde.x_left, pde.x_right, xlen)
t_grid = np.linspace(0, 0.1, tlen)
pde.visual(sol, x_grid, t_grid)