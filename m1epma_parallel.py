from multiprocessing import Process, Pipe
import numpy as np

import sys

sys.path.append('../../m1epma')
sys.path.append('m1epma')
import experiment

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

## Reconstruction definitions:
def build_optimization_functions(true_parameters, true_k_ratios, variable_parameters, x_rays, procs, n_k_ratios):
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
    return f, F, f_and_J
