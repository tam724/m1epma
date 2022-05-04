import numpy as np
from scipy.optimize import least_squares

import sys
sys.path.append('../../m1epma')
sys.path.append('m1epma')
import physics
import experiment
from physics import keV, nano
import optimization
import m1model
import pickle
import k_ratio_model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import m1epma_parallel

elements = [physics.Iron(), physics.Manganese(), physics.Aluminium()]
x_rays = [physics.XRay(e, 1.) for e in elements]

material = physics.Material(
    n_x = 21,
    n_y = 1,
    hat_n_x = 252,
    hat_n_y = 126,
    dim_x = [-525.*nano, 525.*nano],
    dim_y = [-525.*nano, 0.]
)
layer_interfaces = np.linspace(material.dim_x[0], material.dim_x[1], material.n_x + 1)
assert(np.any(np.isclose(layer_interfaces / nano, 25)) or np.any(np.isclose(layer_interfaces / nano, -25)))

detector = physics.Detector(
    x=1.,
    y=1.*np.tan(0.6981),
    material = material)


eps_initial_keV = 10.5
eps_cutoff_keV = 4.5
n_epsilon = 400

beam_energy_keV = 10.

## BEAM VARIANCE -> 50.nano std_dev means, that ~94 percent of the beams energy is inside of 4*50nano (meters)
beam_size_x = (50.*nano)**2

electron_beam = physics.ElectronBeam(
        size=[beam_size_x, None],
        pos=[0., None],
        beam_energy_keV=beam_energy_keV,
        energy_variation_keV=(0.2)**2
    )

e = experiment.Experiment(
        material=material,
        detector=detector,
        electron_beam=electron_beam,
        elements=elements,
        x_ray_transitions=x_rays,
        epsilon_initial_keV=eps_initial_keV,
        epsilon_cutoff_keV=eps_cutoff_keV,
        n_epsilon=n_epsilon
)
## compute standards
true_parameters = np.zeros(e.parameter_dimensions)
true_parameters[:, :, 0] = 1.0
true_parameters[10, :, 0] = 0.0
true_parameters[10, :, 1] = 0.6

std_parameters_MnAl = np.zeros(e.parameter_dimensions)
std_parameters_MnAl[:, :, 1] = 0.8
std_parameters_Fe = np.zeros(e.parameter_dimensions)
std_parameters_Fe[:, :, 0] = 1.0

n_k_ratios = e.n_x_ray_transitions

# k_rat_std = [float(k_ratios_std[1][0]), float(k_ratios_std[0][1]), float(k_ratios_std[0][2])]
procs = m1epma_parallel.start_experiment_processes([e], [1.0, 1.0, 1.0])
i_std_MnTi = m1epma_parallel.compute_k_ratios_async(n_k_ratios, [std_parameters_MnAl], 3, procs, True)

procs = m1epma_parallel.start_experiment_processes([e], [1.0, 1.0, 1.0])
i_std_Fe = m1epma_parallel.compute_k_ratios_async(n_k_ratios, [std_parameters_Fe], 3, procs, True)

procs = m1epma_parallel.start_experiment_processes([e], [i_std_Fe[0], i_std_MnTi[1], i_std_MnTi[2]])
true_k_ratios = m1epma_parallel.compute_k_ratios_async(n_k_ratios, [true_parameters], 3, procs, False)

variable_parameters = [(10, 0, 1)]

def f(x):
    global f_call_count
    parameters = np.copy(true_parameters)
    for i, idx in enumerate(variable_parameters):
        parameters[idx] = x[i]
    k_ratios = m1epma_parallel.compute_k_ratios_async(len(x_rays), [parameters for i in range(len(procs))], n_k_ratios, procs, False)
    # opt_save.append((x, k_ratios - true_k_ratios))
    print("optimization step {}", flush=True)
    print("val", flush=True)
    print(k_ratios - true_k_ratios, flush=True)
    print("p", flush=True)
    print(x, flush=True)
    return k_ratios[1:2] - true_k_ratios[1:2]

def F(x):
    return 1./2. * np.linalg.norm(f(x))

def f_and_J(x):
    print("calculating jacobian")
    parameters = np.copy(true_parameters)
    for i, idx in enumerate(variable_parameters):
        parameters[idx] = x[i]
    k_ratios, k_ratios_jac = m1epma_parallel.compute_k_ratios_jacobian_async(len(x_rays), [parameters for i in range(len(procs))], n_k_ratios, procs, variable_parameters, False)
    return k_ratios[1:2] - true_k_ratios[1:2], k_ratios_jac[1:2, :]

def callback(p, val, step):
    print("optimization step {}".format(step))
    print("val") # value of f
    print(val)
    print("p") # current parameter values
    print(p)
    opt_save.append((p, val))

opt_save = []

x0 = np.full(len(variable_parameters), 0.5) # start value
opt_save.append((x0, 0))
sol = optimization.levenberg_marquard(F, None, None, x0, 10, eps_1=1.e-3, eps_2=1.e-3, f_and_J=f_and_J, callback=callback)