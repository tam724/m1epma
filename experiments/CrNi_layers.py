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
import matplotlib.pyplot as plt

elements = [physics.Chromium(), physics.Nickel()]
x_rays = [physics.XRay(e, 1.) for e in elements]

material = physics.Material(
    n_x = 10,
    n_y = 1,
    hat_n_x = 80,
    hat_n_y = 60,
    dim_x = [0., 1000.*nano],
    dim_y = [-800.*nano, 0.]
)

detector = physics.Detector(
    x=500.*nano,
    y=50.*nano,
    material = material)

eps_initial_keV = 10.5
eps_cutoff_keV = 5.5
n_epsilon = 100

# electron_beams = [
#     physics.ElectronBeam(
#         size=[(30.*nano)**2, (30.*nano)**2],
#         pos=[550.*nano, -100.*nano],
#         beam_energy_keV=12.,
#         energy_variation_keV=0.1
#     ),
#     physics.ElectronBeam(
#         size=[(30.*nano)**2, (30.*nano)**2],
#         pos=[550.*nano, -100.*nano],
#         beam_energy_keV=12.,
#         energy_variation_keV=0.1
#     )
# ]

electron_beams = [
   physics.ElectronBeam(
       size=[(30.*nano)**2, (30.*nano)**2],
       pos=[250.*nano, -100.*nano],
       beam_energy_keV=10.,
       energy_variation_keV=0.1
   ),
   physics.ElectronBeam(
       size=[(30.*nano)**2, (30.*nano)**2],
       pos=[350.*nano, -100.*nano],
       beam_energy_keV=10.,
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
       beam_energy_keV=10.,
       energy_variation_keV=0.1
   ),
   physics.ElectronBeam(
       size=[(30.*nano)**2, (30.*nano)**2],
       pos=[650.*nano, -100.*nano],
       beam_energy_keV=10.,
       energy_variation_keV=0.1
   ),
   physics.ElectronBeam(
       size=[(30.*nano)**2, (30.*nano)**2],
       pos=[750.*nano, -100.*nano],
       beam_energy_keV=10.,
       energy_variation_keV=0.1
   )
]

experiments = [
    experiment.Experiment(
        material=material,
        detector=detector,
        electron_beam=electron_beam,
        elements=elements,
        x_ray_transitions=x_rays,
        epsilon_initial_keV=eps_initial_keV,
        epsilon_cutoff_keV=eps_cutoff_keV,
        n_epsilon=n_epsilon
    )
    for electron_beam in electron_beams]
n_x_ray_transitions = experiments[0].n_x_ray_transitions

# generate the measured k-ratios
true_parameters = np.array(
    [
        np.random.rand(10)
    ]
).reshape((material.n_x, material.n_y, 1))
true_parameters[0, :, :] = 0.5
true_parameters[1, :, :] = 0.5
true_parameters[-2, :, :] = 0.5
true_parameters[-1, :, :] = 0.5
n_parameters = material.n_x*material.n_y

n_k_ratios = sum([e.n_x_ray_transitions for e in experiments])


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
            'epsilon_cutoff_keV': eps_cutoff_keV,
            'epsilon_initial_keV': eps_initial_keV,
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
# variable_parameters = [(2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0)]
variable_parameters = [(2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0)]

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
            'epsilon_cutoff_keV': eps_cutoff_keV,
            'epsilon_initial_keV': eps_initial_keV,
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
with open('CrNi_layers.pkl', 'wb') as writefile:
    pickle.dump((opt_save, sol, true_parameters, true_k_ratios), writefile)