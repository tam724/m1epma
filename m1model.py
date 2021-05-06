from __future__ import absolute_import
from clawpack import pyclaw
import numpy as np
from typing import Tuple

import rpn2_riemann_solver_forward as rpn2_riemann_solver_forward
import k_ratio_model as k_ratio_model
from physics import keV


BEAM_MODEL_INITIAL = 'initial'
BEAM_MODEL_BOUNDARY = 'boundary'

FLUX_PRECISION = 1e-20
ORDER = 1
N_BEAM_STEPS = 100
BEAM_MODEL = BEAM_MODEL_BOUNDARY


def set_beam_model(model):
    global BEAM_MODEL
    if not (model == BEAM_MODEL_INITIAL or model == BEAM_MODEL_BOUNDARY):
        raise ValueError("beam model must be either 'boundary' or 'initial'")
    BEAM_MODEL = model

def zero_boundary_condition_upper(state, dim, t, qbc, auxbc, num_ghost):
    if dim.name == 'x':
        qbc[:, :, -num_ghost:] = 0.0
    elif dim.name == 'y':
        qbc[:, -num_ghost:, :] = 0.0

def zero_boundary_condition_lower(state, dim, t, qbc, auxbc, num_ghost):
    if dim.name == 'x':
        qbc[:, :, :num_ghost] = 0.0
    elif dim.name == 'y':
        qbc[:, :num_ghost, :] = 0.0
    pass

def beam_boundary_condition_upper(state, dim, t, qbc, auxbc, num_ghost, experiment):
    if dim.name == 'x':
        epsilon = (experiment.epsilon_initial_keV - t)
        # print((experiment.epsilon_initial_keV - t))
        x = dim.centers_with_ghost(num_ghost)
        beam = experiment.electron_beam
        #experiment
        #stopping_power = np.zeros(dim.num_cells + num_ghost*2)
        #stopping_power[num_ghost:-num_ghost] = state.aux[0, :, 0]
        #stopping_power[:num_ghost] = stopping_power[num_ghost]
        #stopping_power[-num_ghost:] = stopping_power[-num_ghost-1]
        #experiment
        boundary = np.array([beam.intensity_dist(x_, beam.position_y, epsilon) for x_ in x]) # *stopping_power
        qbc[0, :, -num_ghost:] = np.stack((boundary, boundary), axis=1)
        qbc[1, :, -num_ghost:] = 0.0
        qbc[2, :, -num_ghost:] = -0.9*np.stack((boundary, boundary), axis=1)
    elif dim.name == 'y':
        qbc[:, -num_ghost:, :] = 0.0
    return qbc

def beam_initialize_state(electron_beam, state):
    x, y = state.grid.p_centers
    state.q[0, :, :] = np.exp(-np.power(1 / electron_beam.size_x * (x - electron_beam.position_x), 2) - np.power(1 / electron_beam.size_y * (y - electron_beam.position_y), 2))
    state.q[1, :, :] = 0.0
    state.q[2, :, :] = -0.9 * np.exp(-np.power(1 / electron_beam.size_x * (x - electron_beam.position_x), 2) - np.power(1 / electron_beam.size_y * (y - electron_beam.position_y), 2))

def zero_initialize_state(state):
    state.q[0, :, :] = 0.0
    state.q[1, :, :] = 0.0
    state.q[2, :, :] = 0.0

# ### GENERAL SOLVER SETTINGS ### #
def setup_claw_solver(solver, dt):
    solver.num_eqn = 3
    solver.num_waves = 3
    # setting up aux boundary conditions
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.extrap
    # setting up limiters
    solver.limiters = pyclaw.limiters.tvd.minmod
    # setting up cfl values
    solver.cfl_max = 1.0
    solver.cfl_desired = 0.9
    # setting up initial dt
    solver.dt_initial = dt
    # dt is fixed for all timesteps
    solver.dt_variable = False
    # setting up order
    solver.order = ORDER
    # setting up solver parameters
    solver.source_split = ORDER
    solver.dimensional_split = True
    return solver

def setup_domain(material):
    x = pyclaw.Dimension(material.dim_x[0], material.dim_x[1], material.hat_n_x, name="x")
    y = pyclaw.Dimension(material.dim_y[0], material.dim_y[1], material.hat_n_y, name="y")
    return pyclaw.Domain([x, y])

def setup_state(state):
    state.problem_data['flux_precision'] = FLUX_PRECISION
    state.index_capa = 0
    return state

def setup_controller(state, domain, solver, tfinal, n_epsilon):
    claw = pyclaw.Controller()
    claw.verbosity = 0
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.output_format = None
    claw.outdir = None
    claw.output_style = 2
    claw.out_times = np.linspace(0, claw.tfinal, n_epsilon)
    claw.keep_copy = True
    return claw

class Counter:
    def __init__(self):
        self._i = 0
    
    @property
    def i(self):
        return self._i

    @property
    def reverse_i(self):
        return -(self._i + 1)
    
    def increase(self):
        self._i += 1


# ### FORWARD SOLVER SETTINGS ### #
def setup_forward(experiment, mass_fractions: np.ndarray):
    solver = pyclaw.ClawSolver2D(rpn2_riemann_solver_forward)
    solver = setup_claw_solver(solver, dt=experiment.delta_epsilon_keV)
    # Adding additional parameters
    solver.bc_lower[0] = pyclaw.BC.custom
    solver.bc_lower[1] = pyclaw.BC.custom
    solver.bc_upper[0] = pyclaw.BC.custom
    solver.bc_upper[1] = pyclaw.BC.custom
    solver.user_bc_lower = lambda state, dim, t, qbc, auxbc, num_ghost: zero_boundary_condition_lower(state, dim, t, qbc, auxbc, num_ghost)
    if BEAM_MODEL == BEAM_MODEL_BOUNDARY:
        solver.user_bc_upper = lambda state, dim, t, qbc, auxbc, num_ghost: beam_boundary_condition_upper(state, dim, t, qbc, auxbc, num_ghost, experiment)
    elif BEAM_MODEL == BEAM_MODEL_INITIAL:
        solver.user_bc_upper = lambda state, dim, t, qbc, auxbc, num_ghost: zero_boundary_condition_upper(state, dim, t, qbc, auxbc, num_ghost)

    domain = setup_domain(experiment.material)
    state = pyclaw.State(domain, solver.num_eqn, num_aux=3)
    state = setup_state(state)
    if BEAM_MODEL == BEAM_MODEL_BOUNDARY:
        zero_initialize_state(state)
    elif BEAM_MODEL == BEAM_MODEL_INITIAL:
        beam_initialize_state(experiment.electron_beam, state)

    counter = Counter()
    densities = k_ratio_model.densities(mass_fractions, experiment.specific_densities)
    ext = np.ones(experiment.material.number_of_cells_per_subdomain)
    solver.step_source = src2
    solver.before_step = lambda solver, state: b4step(solver, state, experiment, mass_fractions, densities, counter, ext)
    t_final = experiment.epsilon_initial_keV - experiment.epsilon_cutoff_keV
    claw = setup_controller(state, domain, solver, t_final, experiment.n_epsilon)
    # if BEAM_MODEL == BEAM_MODEL_BOUNDARY:
    #     claw.output_style = 2
    #     el_beam = sim_params.electron_beam
    #     claw.out_times = np.zeros(N_BEAM_STEPS + 1)
    #     claw.out_times[0:N_BEAM_STEPS] = np.linspace(sim_params.energy_to_time(el_beam.beam_energy + 0.7*kilo*electron_volt),
    #                                                  sim_params.energy_to_time(el_beam.beam_energy - 0.7*kilo*electron_volt),
    #                                                  N_BEAM_STEPS)
    #     claw.out_times[N_BEAM_STEPS] = claw.tfinal
    return claw

def src2(solver, state, dt):
    q = state.q
    aux = state.aux
    print(state.t)
    q[0, :, :] = q[0, :, :] + (-aux[1, :, :])*q[0, :, :]*dt / aux[0, :, :]
    q[1, :, :] = q[1, :, :] + (-aux[2, :, :] - aux[1, :, :])*q[1, :, :]*dt / aux[0, :, :]
    q[2, :, :] = q[2, :, :] + (-aux[2, :, :] - aux[1, :, :])*q[2, :, :]*dt / aux[0, :, :]

import matplotlib.pyplot as plt
def b4step(
    solver,
    state,
    experiment,
    mass_fractions : np.ndarray,
    densities: np.ndarray,
    counter: Counter,
    ext: np.ndarray):

    step_count = counter.reverse_i
    # plt.imshow(state.q[0, :, :])
    # plt.show()
    # print(state.t, experiment.epsilon_initial_keV - experiment.epsilons_keV[step_count])
    state.aux[0, :, :] = np.kron(np.einsum('ije,ij,e->ij', mass_fractions, densities, experiment.specific_stopping_power[:, step_count]), ext)/keV
    state.aux[1, :, :] = -np.kron(np.einsum('ije,ij,e->ij', mass_fractions, densities, experiment.specific_stopping_power_d[:, step_count]), ext)
    state.aux[2, :, :] = np.kron(np.einsum('ije,ij,e->ij', mass_fractions, densities, experiment.specific_transport_coefficient[:, step_count]), ext)
    counter.increase()
    # print(state.t)


def solve_forward(
    experiment,
    mass_fractions: np.ndarray):
    
    ex = experiment
    claw = setup_forward(ex, mass_fractions)
    print("calculating forward solution")
    claw.run()
    solution = np.zeros((ex.n_epsilon, 3, ex.material.hat_n_x, ex.material.hat_n_y))
    for i, f in enumerate(reversed(claw.frames)):
        solution[i, :, :, :] = f.state.q
    data = {
        'solution': solution,
        'delta_x': claw.frames[0].delta[0], # TODO: put this in material
        'delta_y': claw.frames[0].delta[1]
    }
    return data
