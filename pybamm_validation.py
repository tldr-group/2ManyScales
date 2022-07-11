import pybamm
import numpy as np
import matplotlib.pyplot as plt
import json

model = pybamm.BaseModel()

n_points = 100
t_points = 1000

# variables
c_e_a = pybamm.Variable("Concentration anode", domain="anode")
c_e_c = pybamm.Variable("Concentration cathode", domain="cathode")
c_e = pybamm.concatenation(c_e_a, c_e_c)

# model.timescale = 1
# model.length_scales = 1

multiplier_a = pybamm.PrimaryBroadcast(1, "anode")
multiplier_c = pybamm.PrimaryBroadcast(-1, "cathode")
multiplier = pybamm.concatenation(multiplier_a, multiplier_c)

# consts
with open('params.json') as f:
    params = json.load(f)
n_points = params['n_points']
t_points = params['t_points']
t_plus = params['t_plus']
c_0 = params['c_0']
# alpha = params['alpha']
alpha = 0.25
PHI = params['PHI']
tau = params['tau']
L = params['L']

# derived params
eps = 1-4*np.pi*alpha**3/3
k_0 = params['j_app']*params['L']*params['F'] / (params['R']*params['T'])
k = params["F"]*params["k_hat"]*c_0**(1/2) / params['j_app']
D_0 = params['j_app']*params['L'] / (params['F']*c_0)
phi_0 = params["R"] * params["T"] / params["F"]
F_0 = params['j_app'] / params['F']
j_0 = params['j_app']

D = 1 * 2.646e-10 / D_0
print(D)
# equations
dcdt = (1/eps) * (pybamm.div(D*pybamm.grad(c_e)) + multiplier * (1-t_plus))

model.rhs[c_e] =  dcdt

# initial conditions
model.initial_conditions = {c_e: pybamm.Scalar(1)}

# bcs
bcs = { c_e: {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
            # "left": (pybamm.Scalar(1), "Dirichlet"),
            # "right": (pybamm.Scalar(0), "Dirichlet"),
        }
}

model.boundary_conditions = bcs
model.variables["Concentration"] =  c_e

#  set param values
#  geometry

x_a = pybamm.SpatialVariable(
    "x_a", domain=["anode"], coord_sys="cartesian"
)
x_c = pybamm.SpatialVariable(
    "x_c", domain=["cathode"], coord_sys="cartesian"
)

geometry = {
    "anode": {x_a: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
    "cathode": {x_c: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
}
submesh_types = {"anode": pybamm.Uniform1DSubMesh, "cathode": pybamm.Uniform1DSubMesh}
var_pts = {x_a: n_points//2, x_c: n_points//2}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {
    "anode": pybamm.FiniteVolume(),
    "cathode": pybamm.FiniteVolume()
}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 10, t_points)
solution = solver.solve(model, t)

c = solution["Concentration"]

def domain_mult(x):
    x = x>1
    x = (x-0.5)*2
    return x


def analytic (x):
    return domain_mult(x) *(1-t_plus) * (x-1)**2 / (2*D)-(1-t_plus)*(x-1)/D+1
# plot
x = np.linspace(0, 2, 100)
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(x*100, c(t=10, x=x)*c_0, label="sim")
plt.plot(x*100, analytic(x)*c_0, '--', label="analytic")
plt.xlabel("x [µm]")
plt.ylabel("Concentration [mol.m-3]")
plt.legend()
plt.tight_layout()
plt.savefig('plots/validation.png')
plt.show()