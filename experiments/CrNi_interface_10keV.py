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

elements = [physics.Chromium(), physics.Nickel()]
x_rays = [physics.XRay(e, 1.) for e in elements]

eps_initial_keV = 10.5
eps_cutoff_keV = 5.5
n_epsilon = 200

def make_material_and_parameters(e_beam_position):
    material = physics.Material(
            n_x = 160,
            n_y = 1,
            hat_n_x = 160,
            hat_n_y = 80,
            dim_x = [e_beam_position-500.*nano, e_beam_position + 500.*nano],
            dim_y = [-500.*nano, 0.]
        )
    print(material.delta_x / nano)
        # compute the layer centers:
    layer_interfaces = np.linspace(material.dim_x[0], material.dim_x[1], material.n_x + 1)
    assert(np.any(np.isclose(layer_interfaces / nano, 0.)))
    layer_centers = (layer_interfaces[1:] + layer_interfaces[:-1])/2.

    params = np.full((material.n_x, material.n_y, 1), 0.0)
    params[:, 0, 0] = 0.8

    for i in range(material.n_x):
        if layer_centers[i] < 0.:
            params[i, 0, 0] = 0.2
    return material, params

def make_material_and_parameters_static():
    material = physics.Material(
            n_x = 2,
            n_y = 1,
            hat_n_x = 160,
            hat_n_y = 80,
            dim_x = [-500.*nano, 500.*nano],
            dim_y = [-500.*nano, 0.]
        )
    print(material.delta_x / nano)
        # compute the layer centers:
    layer_interfaces = np.linspace(material.dim_x[0], material.dim_x[1], material.n_x + 1)
    assert(np.any(np.isclose(layer_interfaces / nano, 0.)))
    layer_centers = (layer_interfaces[1:] + layer_interfaces[:-1])/2.

    params = np.full((material.n_x, material.n_y, 1), 0.0)
    params[:, 0, 0] = 0.8

    for i in range(material.n_x):
        if layer_centers[i] < 0.:
            params[i, 0, 0] = 0.2
    return material, params

def make_experiments_linescan():
    experiments = []
    parameters = []
    for e_beam_position in np.linspace(-500.*nano, 500.*nano, 161):
        material, params = make_material_and_parameters(e_beam_position)
        experiments.append(
            make_experiment(e_beam_position, material))
        parameters.append(params)
    return experiments, parameters

def make_experiments_two_beams():
    experiments = []
    material, true_parameter = make_material_and_parameters_static()
    for e_beam_position in [-100.*nano, 100.*nano]:
        experiments.append(
            make_experiment(e_beam_position, material)
        )
    return experiments, true_parameter

def make_experiments_left_beam():
    experiments = []
    material, true_parameter = make_material_and_parameters_static()
    for e_beam_position in [-100.*nano]:
        experiments.append(
            make_experiment(e_beam_position, material)
        )
    return experiments, true_parameter

def make_experiments_right_beam():
    experiments = []
    material, true_parameter = make_material_and_parameters_static()
    for e_beam_position in [100.*nano]:
        experiments.append(
            make_experiment(e_beam_position, material)
        )
    return experiments, true_parameter

def make_experiments_central_beam():
    experiments = []
    material, true_parameter = make_material_and_parameters_static()
    for e_beam_position in [0.*nano]:
        experiments.append(
            make_experiment(e_beam_position, material)
        )
    return experiments, true_parameter

def make_experiment(e_beam_position, material, compute_internals=False):
    return experiment.Experiment(
                material,
                physics.Detector(
                    x=1.,
                    y=1.*np.tan(0.6981),
                    material = material),
                physics.ElectronBeam(
                        size=[(30.*nano)**2, None],
                        pos=[e_beam_position, None],
                        beam_energy_keV=10.,
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
        if not std_ints:
            e.update_std_intensities()
        else:
            e.update_std_intensities(std_ints)
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
    for exp in experiments:
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
            'n_epsilon': exp.n_epsilon
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

## LINE SCAN
experiments, parameters = make_experiments_linescan()
procs = start_experiment_processes(experiments, [1.0, 1.0])
n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])

true_k_ratios = compute_k_ratios_async(2, parameters, n_k_ratios, procs)
k_r_0 = true_k_ratios[::2]
k_r_1 = true_k_ratios[1::2]

k_ratios_std = compute_std()


## compute std
def compute_std():
    mat, params = make_material_and_parameters(0.)
    params[:] = 1.
    exp = make_experiment(0., mat, True)
    k_ratios_std_0 = experiment.k_ratios(exp, params)

    params[:] = 0.
    k_ratios_std_1 = experiment.k_ratios(exp, params)

    k_ratios_Cr = list(k_r_0 / k_ratios_std_0[0])
    k_ratios_Ni = list(k_r_1 / k_ratios_std_1[1])
    return np.array([k_ratios_Cr, k_ratios_Ni])

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

##Reconstruction runs:
experiments, true_parameters = make_experiments_two_beams()
procs = start_experiment_processes(experiments)
n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])

variable_parameters = [(0, 0, 0), (1, 0, 0)]

true_k_ratios = compute_k_ratios_async(2, [true_parameters for _ in range(len(procs))], n_k_ratios, procs, False)
opt_save = []

x0 = np.full(len(variable_parameters), 0.5) # start value
opt_save.append((x0, 0))
sol = optimization.levenberg_marquard(F, None, None, x0, 10, eps_1=1.e-3, eps_2=1.e-3, f_and_J=f_and_J, callback=callback)


## Reconstruction 2
experiments, true_parameters = make_experiments_right_beam()
procs = start_experiment_processes(experiments)
n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])

variable_parameters = [(0, 0, 0), (1, 0, 0)]

true_k_ratios = compute_k_ratios_async(2, [true_parameters for _ in range(len(procs))], n_k_ratios, procs, False)
opt_save = []

x0 = np.full(len(variable_parameters), 0.5) # start value
opt_save.append((x0, 0))
sol = optimization.levenberg_marquard(F, None, None, x0, 10, eps_1=1.e-3, eps_2=1.e-3, f_and_J=f_and_J, callback=callback)


## Reconstruction 3
experiments, true_parameters = make_experiments_left_beam()
procs = start_experiment_processes(experiments)
n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])

variable_parameters = [(0, 0, 0), (1, 0, 0)]

true_k_ratios = compute_k_ratios_async(2, [true_parameters for _ in range(len(procs))], n_k_ratios, procs, False)
opt_save = []

x0 = np.full(len(variable_parameters), 0.5) # start value
opt_save.append((x0, 0))
sol = optimization.levenberg_marquard(F, None, None, x0, 10, eps_1=1.e-3, eps_2=1.e-3, f_and_J=f_and_J, callback=callback)


## Reconstruction 4
experiments, true_parameters = make_experiments_central_beam()
procs = start_experiment_processes(experiments)
n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])

variable_parameters = [(0, 0, 0), (1, 0, 0)]

true_k_ratios = compute_k_ratios_async(2, [true_parameters for _ in range(len(procs))], n_k_ratios, procs, False)
opt_save = []

x0 = np.full(len(variable_parameters), 0.5) # start value
opt_save.append((x0, 0))
sol = optimization.levenberg_marquard(F, None, None, x0, 11, eps_1=1.e-20, eps_2=1.e-20, f_and_J=f_and_J, callback=callback)

