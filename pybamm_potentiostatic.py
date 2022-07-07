import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel()

n_points = 100
t_points = 1000

# variables
c_e_a = pybamm.Variable("Concentration anode", domain="anode")
c_e_c = pybamm.Variable("Concentration cathode", domain="cathode")
c_e = pybamm.concatenation(c_e_a, c_e_c)

phi_e_a = pybamm.Variable("Potential anode", domain="anode")
phi_e_c = pybamm.Variable("Potential cathode", domain="cathode")
phi_e = pybamm.concatenation(phi_e_a, phi_e_c)

j_e_a = pybamm.Variable("Current density anode", domain="anode")
j_e_c = pybamm.Variable("Current density cathode", domain="cathode")
j_e = pybamm.concatenation(j_e_a, j_e_c)

F_e_a = pybamm.Variable("Flux anode", domain="anode")
F_e_c = pybamm.Variable("Flux cathode", domain="cathode")
F_e = pybamm.concatenation(F_e_a, F_e_c)

multiplier_a = pybamm.PrimaryBroadcast(1, "anode")
multiplier_c = pybamm.PrimaryBroadcast(-1, "cathode")
multiplier = pybamm.concatenation(multiplier_a, multiplier_c)

# consts
F = pybamm.Parameter("Faraday constant [C.mol-1]")
R = pybamm.Parameter("Molar gas constant [J.mol-1.K-1]")
T = pybamm.Parameter("Temperature [K]")
c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
alpha = pybamm.Parameter("Alpha")
L = pybamm.Parameter("Length [m]")
j_app = pybamm.Parameter("Applied current density [A.m-2]")
D0 = pybamm.Parameter("Diffusivity nondim")
tplus = pybamm.Parameter("Transference number")
eps = pybamm.Parameter("Porosity")
PHI = pybamm.Parameter("Applied potential [V]")
k_hat = pybamm.Parameter("BV constant")
k = pybamm.Parameter("BV nondim")
k0 = pybamm.Parameter("Conductivity nondim")

# functions
inputs = {"Concentration": c_e}
D = pybamm.FunctionParameter("Diffusivity", inputs)
K = pybamm.FunctionParameter("Conductivity", inputs)

def diffusion_coeff(c):
    return 5.253e-10 * np.exp( -7.1e-4 * c * c0 ) / D0

def conduction_coeff(c):
    return 1e-4 * c * c0 * ( 5.2069096 - 0.002143628 * c * c0 + 2.34402e-7 * c**2 * c0**2 )**2 / k0


# equations

bv = 4 * np.pi * alpha**2 * k * c_e**(1/2) * ( multiplier * PHI - phi_e ) / 2

dcdt = ( 1 / eps ) * ( -pybamm.grad(F_e) + bv )
model.rhs[c_e] =  dcdt

model.algebraic = {
    F_e: F_e + D * pybamm.grad(c_e) - tplus * j_e,
    j_e: -pybamm.grad(j_e) + bv,
    phi_e: -j_e + K * ( pybamm.grad(phi_e) * 2 * ( 1 - tplus ) * pybamm.grad(c_e) / c_e )
}

# initial conditions
model.initial_conditions = {
    c_e: pybamm.Scalar(1),
    phi_e: pybamm.Scalar(0),
    F_e: pybamm.Scalar(0),
    j_e: pybamm.Scalar(0)
    }

# bcs
bcs = { F_e: {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        },
        j_e: {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }
}

model.boundary_conditions = bcs
model.variables["Concentration"] =  c_e
model.variables["Potential"] =  phi_e
model.variables["Flux"] =  F_e
model.variables["Current density"] =  j_e


#  set param values
param = pybamm.ParameterValues(
    {
        "Faraday constant [C.mol-1]": 96485.33212,
        "Molar gas constant [J.mol-1.K-1]": 8.314,
        "Temperature [K]": 298,
        "Initial concentration [mol.m-3]": 1000,
        "Alpha": 0.37,
        "Length [m]": 1e-4,
        "Applied current density [A.m-2]": 1200,
        "Diffusivity nondim": j_app*L/(F*c0),
        "Transference number": 0.3,
        "Porosity": 1- 4*np.pi*alpha**3/3,
        "Applied potential [V]": 0.05,
        "BV constant": 98.0392e-6,
        "BV nondim": F * k_hat * c0**(1/2) / j_app,
        "Conductivity nondim": j_app*L*F/(R*T),
        "Diffusivity": diffusion_coeff,
        "Conductivity": conduction_coeff,
    }
)

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


# def analytic (x):
#     return domain_mult(x) *(1-tplus) * (x-1)**2 / (2*D)-(1-tplus)*(x-1)/D+1

# plot
x = np.linspace(0, 2, 100)
plt.plot(x, c(t=10, x=x)*c0, label="sim")
# plt.plot(x, analytic(x)*c_0, label="analytic")
plt.xlabel("x")
plt.ylabel("Concentration at t=0.5")
plt.legend()
plt.tight_layout()
plt.show()