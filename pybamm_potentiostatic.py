import pybamm
import numpy as np
import matplotlib.pyplot as plt
import json

model = pybamm.BaseModel()



# variables
c_e_a = pybamm.Variable("Concentration anode", domain="anode")
c_e_c = pybamm.Variable("Concentration cathode", domain="cathode")
c_e = pybamm.concatenation(c_e_a, c_e_c)

phi_e_a = pybamm.Variable("Potential anode", domain="anode")
phi_e_c = pybamm.Variable("Potential cathode", domain="cathode")
phi_e = pybamm.concatenation(phi_e_a, phi_e_c)

multiplier_a = pybamm.PrimaryBroadcast(1, "anode")
multiplier_c = pybamm.PrimaryBroadcast(-1, "cathode")
multiplier = pybamm.concatenation(multiplier_a, multiplier_c)

# params
with open('params.json') as f:
    params = json.loads(f)
n_points = params['n_points']
t_points = params['t_points']
t_plus = params['t_plus']
c_0 = params['c_0']
alpha = params['alpha']
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

def diffusion_coeff(c):
    return 5.253e-10 * pybamm.Exp( -7.1e-4 * c * 1000 ) * eps / ( D_0 * tau )

def conduction_coeff(c):
    return 1e-4 * c * c_0 * ( 5.2069096 - 0.002143628 * c * c_0 + \
        2.34402e-7 * c**2 * c_0**2 )**2 * eps / ( k_0 * tau )


# equations

bv = 4 * np.pi * alpha**2 * k * c_e**(1/2) * ( multiplier * PHI - phi_e ) / 2

dcdt = ( 1 / eps ) * ( -pybamm.div( -diffusion_coeff( c_e ) * pybamm.grad( c_e ) + \
    t_plus * ( -conduction_coeff( c_e ) * ( pybamm.grad( phi_e ) - 2 * (1 - t_plus) * \
    pybamm.grad(c_e) / c_e))) + bv )
model.rhs[c_e] =  dcdt

model.algebraic = {
    phi_e: pybamm.div( -diffusion_coeff( c_e ) * ( pybamm.grad( phi_e ) - 2 * \
        ( 1 - t_plus ) * pybamm.grad( c_e ) / c_e)) - bv
}

#  Derived variables

j_e = -conduction_coeff(c_e) * (pybamm.grad(phi_e) - 2 * (1 - t_plus) * pybamm.grad(c_e) / c_e)
F_e = -diffusion_coeff(c_e) * pybamm.grad(c_e) + t_plus * j_e

# initial conditions
model.initial_conditions = {
    c_e: pybamm.Scalar(1),
    phi_e: pybamm.Scalar(0),
    }

bcs = {
    c_e: {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        },
    phi_e: {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        },
}

model.boundary_conditions = bcs
model.variables["Concentration"] =  c_e
model.variables["Potential"] =  phi_e
model.variables["Flux"] =  F_e
model.variables["Current density"] =  j_e


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
solver = pybamm.CasadiSolver()
t = np.linspace(0, 3, t_points)
solution = solver.solve(model, t)

c = solution["Concentration"]
phi = solution["Potential"]
F = solution["Flux"]
j = solution["Current density"]

def domain_mult(x):
    x = x>1
    x = (x-0.5)*2
    return x


# plot
x = np.linspace(0, 2, 100)
t = 3
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 13))
fig.patch.set_facecolor('white')
ax[0,0].plot(x*100, c(t=t, x=x)*c_0, label="sim")
# plt.plot(x, analytic(x)*c_0, label="analytic")
ax[0,0].set_xlabel("x [µm]")
ax[0,0].set_ylabel("Concentration [mol.m-3]")

ax[0,1].plot(x*100, phi(t=t, x=x)*phi_0, label='sim')
ax[0,1].set_xlabel("x [µm]")
ax[0,1].set_ylabel("Potential [V]")

ax[1,0].plot(x*100, F(t=t, x=x)*F_0, label='sim')
ax[1,0].set_xlabel("x [µm]")
ax[1,0].set_ylabel("Flux [mol.s-1,m-2]")

ax[1,1].plot(x*100, j(t=t, x=x)*j_0, label='sim')
ax[1,1].set_xlabel("x [µm]")
ax[1,1].set_ylabel("Current density [A.m-2]")
plt.legend()
plt.tight_layout()
plt.savefig('plots/potentiostatic_c_phi_F_j.png', transparency=False)
plt.show()
