import numpy as np
from scipy.optimize import least_squares

import sys

from sympy import true
sys.path.append('../../m1epma')
sys.path.append('m1epma')
import physics
import experiment
from physics import keV, nano
import optimization
import m1model
import pickle
import matplotlib.pyplot as plt

elements = [physics.Nickel(), physics.Aluminium()]
x_rays = [physics.XRay(e, 1.) for e in elements]

eps_initial_keV = 15.5
eps_cutoff_keV = 1.5
n_epsilon = 250

def make_material_and_parameters_small_particle(submerged = False):
    material = physics.Material(
        n_x = 32,
        n_y = 16,
        hat_n_x = 192,
        hat_n_y = 96,
        dim_x = [-800.*nano, 800.*nano],
        dim_y = [-800.*nano, 0.]
    )
    print(material.delta_x / nano)
        # compute the layer centers:
    layer_interfaces = np.linspace(material.dim_x[0], material.dim_x[1], material.n_x + 1)
    assert(np.any(np.isclose(layer_interfaces / nano, 0.)))
    layer_centers = (layer_interfaces[1:] + layer_interfaces[:-1])/2.

    params = np.full((material.n_x, material.n_y, 1), 0.0)
    params[:, :, 0] = 0.951 #Matrix (Ni9Al) 95.1% -> c_Al = 4.9%
    
    if submerged:
        params[15, -2, 0] = 0.845
        params[16, -2, 0] = 0.884
        params[15, -3, 0] = 0.884
        params[16, -3, 0] = 0.845
    else:
        params[15, -1, 0] = 0.845
        params[16, -1, 0] = 0.884
        params[15, -2, 0] = 0.884
        params[16, -2, 0] = 0.845
    return material, params

def make_experiments_small_particle(submerged=False):
    experiments = []
    material, true_parameter = make_material_and_parameters_small_particle(submerged)
    for e_beam_position in [-75.*nano, -25.*nano, 25.*nano, 75.*nano]:
        for e_beam_energy in [10., 15.]:
            experiments.append(make_experiment(e_beam_position, material, compute_internals=False, e_beam_energy=e_beam_energy))
    return experiments, true_parameter

def make_experiment(e_beam_position, material, compute_internals=False, e_beam_energy=15.):
    return experiment.Experiment(
                material,
                physics.Detector(
                    x=1.,
                    y=1.*np.tan(0.6981),
                    material = material),
                physics.ElectronBeam(
                        size=[(30.*nano)**2, None],
                        pos=[e_beam_position, None],
                        beam_energy_keV=e_beam_energy,
                        energy_variation_keV=0.2**2.
                    ),
                elements,
                x_rays,
                eps_initial_keV,
                eps_cutoff_keV,
                n_epsilon,
                compute_internals=compute_internals
            )


from multiprocessing import Process, Pipe

def start_experiment_processes(experiments, std_ints=None):
    if not std_ints:
        std_ints = [None for e in experiments]
    def worker(conn):
        conn.poll(None)
        setup = conn.recv()
        e = experiment.Experiment(
            material=setup['material'],
            detector=setup['detector'],
            electron_beam=setup['electron_beam'],
            elements=setup['elements'],
            x_ray_transitions=setup['x_ray_transitions'],
            epsilon_cutoff_keV=setup['epsilon_cutoff_keV'],
            epsilon_initial_keV=setup['epsilon_initial_keV'],
            n_epsilon=setup['n_epsilon']
        )
        if not setup['std_int']:
            e.update_std_intensities()
        else:
            e.update_std_intensities(setup['std_int'])
        while True:
            conn.poll(None)
            work = conn.recv()
            if(work['what'] == 'k_ratios'):
                k_ratios = experiment.k_ratios(e, work['parameters'])
                conn.send(k_ratios)
            elif(work['what'] == 'k_ratios_jacobian'):
                k_ratios, k_ratios_jac = experiment.k_ratios_jacobian(e, work['parameters'])
                conn.send((k_ratios, k_ratios_jac))
            else:
                print("ERROR")
    # actual function
    procs = []
    for (exp, std) in zip(experiments, std_ints):
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(child_conn, ))
        p.start()
        parent_conn.send(
        {
            'material': exp.material,
            'detector': exp.detector,
            'electron_beam': exp.electron_beam,
            'elements': exp.elements,
            'x_ray_transitions': exp.x_ray_transitions,
            'epsilon_cutoff_keV': exp.epsilon_cutoff_keV,
            'epsilon_initial_keV': exp.epsilon_initial_keV,
            'n_epsilon': exp.n_epsilon,
            'std_int': std
        })
        procs.append((p, parent_conn))
    return procs

def compute_k_ratios_async(n_x_ray_transitions, parameters, n_k_ratios, procs, kill_afterwards=True):
    k_ratios = np.zeros(n_k_ratios)
    i_problem = 0
    while i_problem <= len(procs):
        for i in range(min(4, len(procs) - i_problem)):
            _, conn = procs[i_problem + i]
            params = parameters[i_problem + i]
            work = {
                'what': 'k_ratios',
                'parameters': params
            }
            conn.send(work)
        for i in range(min(4, len(procs) - i_problem)):
            p, conn = procs[i_problem + i]
            conn.poll(None)
            k_ratios_e = conn.recv()
            k_ratios[(i_problem + i)*n_x_ray_transitions:(i_problem + i + 1)*n_x_ray_transitions] = k_ratios_e
            if kill_afterwards:
                p.terminate()
        i_problem += 4
    return k_ratios

def compute_k_ratios_jacobian_async(n_x_ray_transitions, parameters, n_k_ratios, procs, variable_parameters, kill_afterwards=True):
    k_ratios = np.zeros(n_k_ratios)
    k_ratios_jac = np.zeros((n_k_ratios, len(variable_parameters)))
    i_problem = 0
    while i_problem <= len(procs):
        for i in range(min(4, len(procs) - i_problem)):
            _, conn = procs[i_problem + i]
            params = parameters[i_problem + i]
            work = {
                'what': 'k_ratios_jacobian',
                'parameters': params
            }
            conn.send(work)
        for i in range(min(4, len(procs) - i_problem)):
            p, conn = procs[i_problem + i]
            conn.poll(None)
            k_ratios_e, k_ratios_jac_e = conn.recv()
            k_ratios[(i_problem + i)*n_x_ray_transitions:(i_problem + i + 1)*n_x_ray_transitions] = k_ratios_e
            for j in range(n_x_ray_transitions):
                for k, idx in enumerate(variable_parameters):
                    k_ratios_jac[(i_problem + i)*n_x_ray_transitions+j, k] = k_ratios_jac_e[(j,) + idx]
            if kill_afterwards:
                p.terminate()
        i_problem += 4
    return k_ratios, k_ratios_jac

    # k_ratios_jac = np.zeros((n_k_ratios, len(variable_parameters)))
    # work = {
    #     'what': 'k_ratios_jacobian',
    #     'parameters': parameters
    # }
    # for (_, conn) in procs:
    #     conn.send(work)
    # for i, (_, conn) in enumerate(procs):
    #     conn.poll(None)
    #     k_ratios_e, k_ratios_jac_e = conn.recv()
    #     k_ratios[i*n_x_ray_transitions:(i+1)*n_x_ray_transitions] = k_ratios_e
    #     for j in range(n_x_ray_transitions):
    #         for k, idx in enumerate(variable_parameters):
    #             k_ratios_jac[i*n_x_ray_transitions+j, k] = k_ratios_jac_e[(j,)+ idx]
    # return k_ratios - true_k_ratios, k_ratios_jac

## Reconstruction definitions:
def f(x):
    global f_call_count
    parameters = np.copy(true_parameters)
    for i, idx in enumerate(variable_parameters):
        parameters[idx] = x[i]
    k_ratios = compute_k_ratios_async(len(x_rays), [parameters for i in range(len(procs))], n_k_ratios, procs, False)
    # opt_save.append((x, k_ratios - true_k_ratios))
    print("optimization step {}", flush=True)
    print("val", flush=True)
    print(k_ratios - true_k_ratios, flush=True)
    print("p", flush=True)
    print(x, flush=True)
    return k_ratios - true_k_ratios

def F(x):
    return 1./2. * np.linalg.norm(f(x))

def f_and_J(x):
    print("calculating jacobian")
    parameters = np.copy(true_parameters)
    for i, idx in enumerate(variable_parameters):
        parameters[idx] = x[i]
    k_ratios, k_ratios_jac = compute_k_ratios_jacobian_async(len(x_rays), [parameters for i in range(len(procs))], n_k_ratios, procs, variable_parameters, False)
    return k_ratios - true_k_ratios, k_ratios_jac

def callback(p, val, step):
    print("optimization step {}".format(step))
    print("val") # value of f
    print(val)
    print("p") # current parameter values
    print(p)
    opt_save.append((p, val))


## Reconstruction 5

## TESTING
# mat, par = make_material_and_parameters_small_particle()
# exp = make_experiment(0., mat, True)

# true_parameters = np.zeros(exp.parameter_dimensions)
# true_parameters[:, :, 0] = 0.867 # for the standard
# mass_fractions = experiment.mass_fractions_from_parameters(mat.n_x, mat.n_y, true_parameters)
# test = m1model.solve_forward(exp, mass_fractions)

# plt.imshow(test['solution'][150, 0, :, :]); plt.show();

experiments, true_parameters = make_experiments_small_particle(submerged=True)
procs = start_experiment_processes(experiments, [[1.0, 1.0] for _ in experiments])

## COMPUTE STANDARD
par = np.zeros(experiments[0].parameter_dimensions)
par[:, :, 0] = 0.867 # for the standard
n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])
k_ratios_std = compute_k_ratios_async(2, [par for _ in experiments], n_k_ratios, procs, True)

procs = start_experiment_processes(experiments, [[k_ratios_std[0], k_ratios_std[1]] if e.electron_beam.beam_energy_keV == 10. else [k_ratios_std[2], k_ratios_std[3]] for e in experiments])


# non submerged
#variable_parameters = [(14, -1, 0), (15, -1, 0), (16, -1, 0), (17, -1, 0), (14, -2, 0), (15, -2, 0), (16, -2, 0), (17, -2, 0)]
# submerged
variable_parameters = [(14, -2, 0), (15, -2, 0), (16, -2, 0), (17, -2, 0), (14, -3, 0), (15, -3, 0), (16, -3, 0), (17, -3, 0)]

true_k_ratios = compute_k_ratios_async(2, [true_parameters for _ in range(len(procs))], n_k_ratios, procs, False)


# test_parameters = np.copy(true_parameters)
# test_parameters[variable_parameters[0]] = 0.5
# test_parameters[variable_parameters[1]] = 0.5
# test_k_ratios = compute_k_ratios_async(2, [test_parameters for _ in range(len(procs))], n_k_ratios, procs, False)
opt_save = []

x0 = np.full(len(variable_parameters), 0.867) # start value
x0 = np.random.rand(len(variable_parameters))
x0 = 0.6 + np.random.rand(len(variable_parameters))*0.4
opt_save.append((x0, 0))
sol = optimization.levenberg_marquard(F, None, None, x0, 30, eps_1=1.e-7, eps_2=1.e-7, f_and_J=f_and_J, callback=callback)
