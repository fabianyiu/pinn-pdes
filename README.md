# PINNs vs Classical PDE Solvers under Data Scarcity
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-blue)
![PDE](https://img.shields.io/badge/Physics-PDE-red)
![Method](https://img.shields.io/badge/Numerical%20Method-Finite%20Difference-orange)
![SciML](https://img.shields.io/badge/Direction-Scientific%20Machine%20Learning-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Description
This project simulates 1D heat flow in a rod (one end heated instantly, one end at room temperature) and compares exact solution, finite-difference and physics-informed neural network (PINN) solvers. Dirichlet boundary conditions fix the hot end at 100°C from t=0 while the rod starts at 24°C, so the setup has a discontinuity at that boundary; the project observes this behaviour and investigates how PINNs handle it. It also studies how the balance of the loss terms—PDE (collocation), initial condition, and boundary conditions—affects PINN accuracy and training behaviour. A Streamlit app provides interactive visualisations of the solutions and loss terms so users can explore the results and demonstrates shipping the study as a simple, shareable web app.
---

## Physics scenario
Sudden turn-on of heat at the right end is modelled: the rod starts at 24°C and the right end is held at 100°C from t=0, giving a discontinuity at (x=1, t=0). 

The discontinuity is forced by the problem setup (IC vs BC at the corner) rather than by a physical defect, but it is still useful: it gives a well-defined test case with a known exact solution, so we can see how classical and PINN solvers handle a sharp jump and related effects (e.g. Gibbs in the Fourier solution) and whether PINNs smooth the discontinuity or capture it.

For the exact solution is a Fourier series, approximated with finitely many modes. Near the discontinuity this leads to Gibbs oscillations; the solution still converges to the correct steady state.

For FDM 

For PINN

## Example Output

*(Insert plots / screenshots of interactive webapp and also results here)*

---

## Heat Equation

The PDE studied in this repository is the **1D heat equation**:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

Boundary and initial conditions are chosen so that an **analytical solution exists**, allowing comparison between:

- Exact solution
- Finite Difference Method (FDM)
- Physics-Informed Neural Networks (PINNs)

---

## How to Run

```bash
git clone https://github.com/yourrepo
cd yourrepo
pip install -r requirements.txt

python train_pinn.py
python run_fdm.py
```
## Discussion and Results


Experimental/Dev notes are located at src/log.md 