from __future__ import absolute_import
from clawpack import pyclaw
import numpy as np
from typing import Callable

import rpn2_riemann_solver_adjoint as rpn2_riemann_solver_adjoint
import m1model as m1model
import k_ratio_model as k_ratio_model
from physics import keV


def setup_adjoint(
    experiment,
    adjoint_source: Callable,
    mass_fractions: np.ndarray,
    forward_solution: np.ndarray):

    solver = pyclaw.ClawSolver2D(rpn2_riemann_solver_adjoint)
    solver = m1model.setup_claw_solver(solver, dt=experiment.delta_epsilon_keV)

    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.extrap
    #solver.user_bc_lower = lambda state, dim, t, qbc, auxbc, num_ghost: m1model.zero_boundary_condition(state, dim, t, qbc, auxbc, num_ghost, sim_params)
    #solver.user_bc_upper = lambda state, dim, t, qbc, auxbc, num_ghost: m1model.zero_boundary_condition(state, dim, t, qbc, auxbc, num_ghost, sim_params)

    domain = m1model.setup_domain(experiment.material)
    # auxiliary variables are (are calculated/set in b4step)
    # 0: stopping_power
    # 1: stopping_power_d
    # 2: transport_coefficient
    # 3,4,5: forward_solution
    # 6: adjoint source
    state = pyclaw.State(domain, num_eqn=solver.num_eqn, num_aux=7)
    state = m1model.setup_state(state)
    m1model.zero_initialize_state(state)

    counter = m1model.Counter()
    densities = k_ratio_model.densities(mass_fractions, experiment.specific_densities)
    ext = np.ones(experiment.material.number_of_cells_per_subdomain)
    solver.step_source = src2
    solver.before_step = lambda solver, state : b4step(solver, state, experiment, adjoint_source, mass_fractions, densities, forward_solution, counter, ext)

    tfinal = experiment.epsilon_initial_keV - experiment.epsilon_cutoff_keV
    claw = m1model.setup_controller(state, domain, solver, tfinal, experiment.n_epsilon)
    return claw

def src2(solver, state, dt):
    # NOTE: aux[1,:,:] is initialized with zeros
    q = state.q
    aux = state.aux

    q[0, :, :] = q[0, :, :] + (aux[6, :, :]) * dt / aux[0, :, :]
    q[1, :, :] = q[1, :, :] + (-aux[2, :, :] * q[1, :, :]) * dt / aux[0, :, :]
    q[2, :, :] = q[2, :, :] + (-aux[2, :, :] * q[2, :, :]) * dt / aux[0, :, :]

def b4step(
    solver,
    state,
    experiment,
    adjoint_source: Callable,
    mass_fractions: np.ndarray,
    densities: np.ndarray,
    forward_solution: np.ndarray,
    counter,
    ext: np.ndarray):

    step_count = counter.i
    # print(state.t, experiment.epsilon_initial_keV - experiment.epsilons_keV[step_count])

    state.aux[3:6, :, :] = forward_solution[step_count, :, :, :]
    state.aux[0, :, :] = np.kron(np.einsum('ije,ij,e->ij', mass_fractions, densities, experiment.specific_stopping_power[:, step_count]), ext)/keV
    state.aux[1, :, :] = 0.0 #-np.kron(np.einsum('ije,ij,e->ij', mass_fractions, densities, experiment.specific_stopping_power_d[:, step_count]), ext)
    state.aux[2, :, :] = np.kron(np.einsum('ije,ij,e->ij', mass_fractions, densities, experiment.specific_transport_coefficient[:, step_count]), ext)
    state.aux[6, :, :] = adjoint_source(counter)
    
    counter.increase()
    # print(state.t)


def solve_adjoint(
    experiment,
    adjoint_source: Callable,
    mass_fractions: np.ndarray,
    forward_solution: np.ndarray):

    ex = experiment
    claw = setup_adjoint(experiment, adjoint_source, mass_fractions, forward_solution)
    print("calculating adjoint solution")
    claw.run()
    solution = np.zeros((ex.n_epsilon, 3, ex.material.hat_n_x, ex.material.hat_n_y))
    for i, f in enumerate(claw.frames):
        solution[i, :, :, :] = f.state.q
    data = {
        'solution': solution,
        'delta_x': claw.frames[0].delta[0], # TODO: put this in material
        'delta_y': claw.frames[0].delta[1]
    }
    return data