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


elements = [physics.Copper(), physics.Manganese()]
x_rays = [physics.XRay(e, 1.) for e in elements]

material = physics.Material(
    n_x = 10,
    n_y = 10,
    hat_n_x = 160,
    hat_n_y = 120,
    dim_x = [0., 1000.*nano],
    dim_y = [-800.*nano, 0.]
)

material_high_res = physics.Material(
    n_x = 10,
    n_y = 10,
    hat_n_x = 160,
    hat_n_y = 120,
    dim_x = [0., 1000.*nano],
    dim_y = [-800.*nano, 0.]
)

detector = physics.Detector(
    x=500.*nano,
    y=50.*nano,
    material = material)

detector_high_res = physics.Detector(
    x=500.*nano,
    y=50.*nano,
    material = material_high_res)

electron_beams = [
    # physics.ElectronBeam(
    #     size=[(30.*nano)**2, (30.*nano)**2],
    #     pos=[350.*nano, -100.*nano],
    #     beam_energy_keV=12.,
    #     energy_variation_keV=0.1
    # ),
    # physics.ElectronBeam(
    #     size=[(30.*nano)**2, (30.*nano)**2],
    #     pos=[350.*nano, -100.*nano],
    #     beam_energy_keV=11.,
    #     energy_variation_keV=0.1
    # ),
    # physics.ElectronBeam(
    #     size=[(30.*nano)**2, (30.*nano)**2],
    #     pos=[350.*nano, -100.*nano],
    #     beam_energy_keV=10.,
    #     energy_variation_keV=0.1
    # ),
    physics.ElectronBeam(
        size=[(30.*nano)**2, (30.*nano)**2],
        pos=[450.*nano, -100.*nano],
        beam_energy_keV=12.,
        energy_variation_keV=0.1
    ),
    physics.ElectronBeam(
        size=[(30.*nano)**2, (30.*nano)**2],
        pos=[450.*nano, -100.*nano],
        beam_energy_keV=11.,
        energy_variation_keV=0.1
    ),
    physics.ElectronBeam(
        size=[(30.*nano)**2, (30.*nano)**2],
        pos=[450.*nano, -100.*nano],
        beam_energy_keV=10.,
        energy_variation_keV=0.1
    ),
    physics.ElectronBeam(
        size=[(30.*nano)**2, (30.*nano)**2],
        pos=[550.*nano, -100.*nano],
        beam_energy_keV=12.,
        energy_variation_keV=0.1
    ),
    physics.ElectronBeam(
        size=[(30.*nano)**2, (30.*nano)**2],
        pos=[550.*nano, -100.*nano],
        beam_energy_keV=11.,
        energy_variation_keV=0.1
    ),
    physics.ElectronBeam(
        size=[(30.*nano)**2, (30.*nano)**2],
        pos=[550.*nano, -100.*nano],
        beam_energy_keV=10.,
        energy_variation_keV=0.1
    ),
    # physics.ElectronBeam(
    #     size=[(30.*nano)**2, (30.*nano)**2],
    #     pos=[650.*nano, -100.*nano],
    #     beam_energy_keV=12.,
    #     energy_variation_keV=0.1
    # ),
    # physics.ElectronBeam(
    #     size=[(30.*nano)**2, (30.*nano)**2],
    #     pos=[650.*nano, -100.*nano],
    #     beam_energy_keV=11.,
    #     energy_variation_keV=0.1
    # ),
    # physics.ElectronBeam(
    #     size=[(30.*nano)**2, (30.*nano)**2],
    #     pos=[650.*nano, -100.*nano],
    #     beam_energy_keV=10.,
    #     energy_variation_keV=0.1
    # )
]

experiments = [
    experiment.Experiment(
        material=material,
        detector=detector,
        electron_beam=electron_beam,
        elements=elements,
        x_ray_transitions=x_rays,
        epsilon_initial_keV=13.,
        epsilon_cutoff_keV=5.,
        n_epsilon=300
    )
    for electron_beam in electron_beams]
n_x_ray_transitions = experiments[0].n_x_ray_transitions

# generate the measured k-ratios
true_parameters = np.array(
    [
        [0.78, 0.72, 0.71, 0.65, 0.2 , 0.09, 0.97, 0.17, 0.62, 0.85],
        [0.03, 0.16, 0.78, 0.28, 0.23, 0.45, 0.09, 0.06, 0.97, 0.67],
        [0.6 , 0.49, 0.09, 0.48, 0.07, 0.27, 0.76, 0.95, 0.03, 0.09],
        [0.96, 0.64, 0.4 , 0.26, 0.8 , 0.93, 0.19, 0.9 , 0.87, 0.08],
        [0.16, 0.9 , 0.5 , 0.83, 0.62, 0.41, 0.19, 0.32, 0.61, 0.36],
        [0.62, 0.75, 0.47, 0.61, 0.68, 0.65, 0.98, 0.35, 0.14, 0.94],
        [0.72, 0.78, 0.96, 0.44, 0.03, 0.66, 0.16, 0.56, 0.72, 0.21],
        [0.  , 0.38, 0.86, 0.55, 0.21, 0.66, 0.49, 0.44, 0.66, 0.11],
        [0.81, 0.19, 0.58, 0.8 , 0.63, 0.77, 0.82, 0.32, 0.97, 0.64],
        [0.48, 0.05, 0.16, 0.56, 0.49, 0.46, 0.58, 0.04, 0.84, 1.  ]
    ]
).reshape((material.n_x, material.n_y, 1))
n_parameters = material.n_x*material.n_y

n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])

# def run_objective_plots():
#     true_k_ratios = np.zeros(n_k_ratios)
#     for i, e in enumerate(experiments):
#         k_ratios_e = experiment.k_ratios(e, true_parameters)
#         true_k_ratios[i*n_x_ray_transitions:(i+1)*n_x_ray_transitions] = k_ratios_e
#     print(true_k_ratios)
#     def f(parameters):
#         k_ratios = np.zeros(n_k_ratios)
#         for i, e in enumerate(experiments):
#             k_ratios_e = experiment.k_ratios(e, parameters)
#             k_ratios[i*n_x_ray_transitions:(i+1)*n_x_ray_transitions] = k_ratios_e
#         return k_ratios - true_k_ratios

#     def F(x):
#         return 1./2. * np.linalg.norm(f(x))

#     N = 40
#     objective1 = np.zeros((N, N))
#     for i, p1 in enumerate(np.linspace(0, 1, N)):
#         for j, p2 in enumerate(np.linspace(0, 1, N)):
#             parame = np.copy(true_parameters)
#             parame[4, -2] = p1
#             parame[5, -2] = p2
#             objective1[i, j] = F(parame)
#     with open('objective1.pkl', 'wb') as writefile:
#         pickle.dump(objective1, writefile)   
    
#     objective2 = np.zeros((N, N))
#     for i, p1 in enumerate(np.linspace(0, 1, N)):
#         for j, p2 in enumerate(np.linspace(0, 1, N)):
#             parame = np.copy(true_parameters)
#             parame[4, -1] = p1
#             parame[4, -2] = p2
#             objective2[i, j] = F(parame)
#     with open('objective2.pkl', 'wb') as writefile:
#         pickle.dump(objective2, writefile) 

from multiprocessing import Process, Pipe

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
    e.update_std_intensities()
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

def run_global_optimization():
    procs = []
    for electron_beam in electron_beams:
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(child_conn, ))
        p.start()
        parent_conn.send(
            {
                'material': material_high_res,
                'detector': detector_high_res,
                'electron_beam': electron_beam,
                'elements': elements,
                'x_ray_transitions': x_rays,
                'epsilon_cutoff_keV': 8.,
                'epsilon_initial_keV': 18.,
                'n_epsilon': 300
            }
        )
        procs.append((p, parent_conn))
    

    true_k_ratios = np.zeros(n_k_ratios)
    work = {
        'what': 'k_ratios',
        'parameters': true_parameters
    }
    for (_, conn) in procs:
        conn.send(work)
    
    for i, (_, conn) in enumerate(procs):
        conn.poll(None)
        k_ratios_e = conn.recv()
        true_k_ratios[i*n_x_ray_transitions:(i+1)*n_x_ray_transitions] = k_ratios_e
    print("true_k_ratios")
    print(true_k_ratios)


    opt_save = []
    # variable_parameters = [(3, -1, 0), (4, -1, 0), (5, -1, 0), (6, -1, 0), (3, -2, 0), (4, -2, 0), (5, -2, 0), (6, -2, 0)]
    variable_parameters = [(4, -1, 0), (5, -1, 0), (4, -2, 0), (5, -2, 0)]

    for (p, conn) in procs:
        p.terminate()
    procs = []
    for electron_beam in electron_beams:
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(child_conn, ))
        p.start()
        parent_conn.send(
            {
                'material': material,
                'detector': detector,
                'electron_beam': electron_beam,
                'elements': elements,
                'x_ray_transitions': x_rays,
                'epsilon_cutoff_keV': 8.,
                'epsilon_initial_keV': 18.,
                'n_epsilon': 300
            }
        )
        procs.append((p, parent_conn))

    def f(x):
        global f_call_count
        parameters = np.copy(true_parameters)
        for i, idx in enumerate(variable_parameters):
            parameters[idx] = x[i]
        k_ratios = np.zeros(n_k_ratios)
        work = {
            'what': 'k_ratios',
            'parameters': parameters
        }
        for (_, conn) in procs:
            conn.send(work)
        for i, (_, conn) in enumerate(procs):
            conn.poll(None)
            k_ratios_e = conn.recv()
            k_ratios[i*n_x_ray_transitions:(i+1)*n_x_ray_transitions] = k_ratios_e
        # opt_save.append((x, k_ratios - true_k_ratios))
        print("optimization step {}", flush=True)
        print("val", flush=True)
        print(k_ratios - true_k_ratios, flush=True)
        print("p", flush=True)
        print(x, flush=True)
        return k_ratios - true_k_ratios

    def F(x):
        return 1./2. * np.linalg.norm(f(x))

    from multiprocessing import Pool
    def f_and_J(x):
        print("calculating jacobian")
        parameters = np.copy(true_parameters)
        for i, idx in enumerate(variable_parameters):
            parameters[idx] = x[i]
        k_ratios = np.zeros(n_k_ratios)
        k_ratios_jac = np.zeros((n_k_ratios, len(variable_parameters)))
        work = {
            'what': 'k_ratios_jacobian',
            'parameters': parameters
        }
        for (_, conn) in procs:
            conn.send(work)
        for i, (_, conn) in enumerate(procs):
            conn.poll(None)
            k_ratios_e, k_ratios_jac_e = conn.recv()
            k_ratios[i*n_x_ray_transitions:(i+1)*n_x_ray_transitions] = k_ratios_e
            for j in range(n_x_ray_transitions):
                for k, idx in enumerate(variable_parameters):
                    k_ratios_jac[i*n_x_ray_transitions+j, k] = k_ratios_jac_e[(j,)+ idx]
        return k_ratios - true_k_ratios, k_ratios_jac

    def callback(p, val, step):
        print("optimization step {}".format(step))
        print("val") # value of f
        print(val)
        print("p") # current parameter values
        print(p)
        opt_save.append((p, val))

    x0 = np.full(len(variable_parameters), 0.5) # start value
    # x0 = np.random.random(len(variable_parameters))
    # x0 = np.array([0.46, 0.54, 0.6, 0.46])
    opt_save.append((x0, 0))
    sol = optimization.levenberg_marquard(F, None, None, x0, 100, eps_1=1.e-25, eps_2=1.e-25, f_and_J=f_and_J, callback=callback)
    # sol = least_squares(fun=f, x0=x0, jac=f_and_J, bounds=(np.zeros((100, 1)), np.ones((100, 1))))

    for (p, conn) in procs:
        p.terminate()
    with open('optimization_plots_same_resolution_160.pkl', 'wb') as writefile:
        pickle.dump((opt_save, sol), writefile)

# def run_m1model_plots():
#     e = experiments[4]
#     mass_fractions = experiment.mass_fractions_from_parameters(e.material.n_x, e.material.n_y, true_parameters)
#     m1model_data = m1model.solve_forward(e, mass_fractions)
#     m1model_data['material'] = e.material
#     m1model_data['beam'] = e.electron_beam
#     m1model_data['detector'] = e.detector
#     m1model_data['energies_keV'] = e.epsilons_keV
#     with open('m1model_plots.pkl', 'wb') as writefile:
#         pickle.dump(m1model_data, writefile)   

if __name__ == "__main__":      
    #run_m1model_plots()
    #run_objective_plots()
    run_global_optimization()