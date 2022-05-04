import numpy as np
import jax.numpy as jnp
from jax import jacrev
import jax
from typing import List, Dict
jax.config.update("jax_enable_x64", True)

import physics
import m1model
import m1model_adjoint
import k_ratio_model
from physics import keV

class Experiment:
    def __init__(
        self, 
        material:physics.Material,
        detector:physics.Detector,
        electron_beam:physics.ElectronBeam,
        elements:List[physics.Element],
        x_ray_transitions:List[physics.XRay],
        epsilon_initial_keV:float,
        epsilon_cutoff_keV:float,
        n_epsilon:int,
        compute_internals=True):

        self._material = material
        self._detector = detector
        self._electron_beam = electron_beam
        self._elements = elements
        self._n_elements = len(elements)
        self._x_ray_transitions = x_ray_transitions
        self._n_x_ray_transitions = len(x_ray_transitions)
        self._epsilon_initial_keV = epsilon_initial_keV
        self._epsilon_cutoff_keV = epsilon_cutoff_keV
        self._n_epsilon = n_epsilon
        self._epsilons_keV, self._delta_epsilon_keV = np.linspace(epsilon_cutoff_keV, epsilon_initial_keV, n_epsilon, retstep=True)

        if compute_internals:
            self._update_specific_densities()
            self._update_attenumation_integration_segments()
            self._update_specific_attenuation_coefficient()
            self._update_standard_intensities()
            self._update_specific_n_of_atoms()
            self._update_emiss_cross_sections()
            self._update_specific_stopping_power_and_transport_coefficient()
            self._update_extraction_operator()

    @property
    def material(self) -> physics.Material:
        return self._material

    # @material.setter
    # def material(self, material : physics.Material):
    #     self._material = material
    #     self._update_attenumation_integration_segments()
    
    @property
    def detector(self) -> physics.Detector:
        return self._detector

    # @detector.setter
    # def detector(self, detector : physics.Detector):
    #     self._detector = detector
    #     self._update_attenumation_integration_segments()

    @property
    def electron_beam(self) -> physics.ElectronBeam:
        return self._electron_beam

    # @electron_beam.setter
    # def exectron_beam(self, electron_beam : physics.ElectronBeam):
    #     self._electron_beam = electron_beam

    @property
    def elements(self) -> List[physics.Element]:
        return self._elements

    # @elements.setter
    # def elements(self, elements: List[physics.Element]):
    #     self._elements = elements
    #     self._n_elements = len(elements)

    #     self._update_specific_densities()
    #     self._update_specific_attenuation_coefficient()
    #     self._update_specific_stopping_power_and_transport_coefficient()

    @property
    def n_elements(self) -> int:
        return self._n_elements

    @property
    def x_ray_transitions(self) -> List[physics.XRay]:
        return self._x_ray_transitions

    # @x_ray_transitions.setter
    # def x_ray_transitions(self, x_ray_transitions: List[physics.XRay]):
    #     self._x_ray_transitions = x_ray_transitions
    #     self._n_x_ray_transitions = len(x_ray_transitions)

    #     self._update_specific_attenuation_coefficient()
    #     self._update_emiss_cross_sections()
    #     self._update_standard_intensities()
    #     self._update_specific_n_of_atoms()

    @property
    def n_x_ray_transitions(self) -> int:
        return self._n_x_ray_transitions

    @property
    def n_epsilon(self) -> int:
        return self._n_epsilon

    @property
    def epsilon_initial_keV(self) -> float:
        return self._epsilon_initial_keV

    @property
    def epsilon_cutoff_keV(self) -> float:
        return self._epsilon_cutoff_keV
    
    @property
    def epsilons_keV(self) -> np.ndarray:
        return self._epsilons_keV

    @property
    def delta_epsilon_keV(self) -> float:
        return self._delta_epsilon_keV

    # def set_energy_interval_keV(self, epsilon_initial, epsilon_cutoff, n_epsilon):
    #     self._epsilon_initial_keV = epsilon_initial
    #     self._epsilon_cutoff_keV = epsilon_cutoff
    #     self._n_epsilon = n_epsilon
    #     self._epsilons_keV, self._delta_epsilon_keV = np.linspace(epsilon_cutoff, epsilon_initial, n_epsilon, retstep=True)

    #     self._update_emiss_cross_sections()
    #     self._update_specific_stopping_power_and_transport_coefficient()
    
    @property
    def epsilon_initial_J(self) -> float:
        return self.epsilon_initial_keV*keV

    @property
    def epsilon_cutoff_J(self) -> float:
        return self.epsilon_cutoff_keV*keV
    
    @property
    def epsilons_J(self) -> float:
        return self.epsilons_keV*keV

    @property
    def delta_epsilon_J(self) -> float:
        return self.delta_epsilon_keV*keV

    @property
    def specific_densities(self) -> np.ndarray:
        return self._specific_densities

    def _update_specific_densities(self):
        self._specific_densities = np.array([e.density() for e in self._elements])

    @property
    def specific_attenuation_coefficients(self) -> np.ndarray:
        return self._specific_attenuation_coefficient
    
    def _update_specific_attenuation_coefficient(self):
        self._specific_attenuation_coefficient = np.zeros((self.n_x_ray_transitions, self.n_elements))
        for i, e in enumerate(self.elements):
            for j, xr in enumerate(self.x_ray_transitions):
                self._specific_attenuation_coefficient[j, i] = e.mass_absorption_coefficient(xr.photon_energy())

    @property
    def emiss_cross_sections(self) -> np.ndarray:
        return self._emiss_cross_sections

    def _update_emiss_cross_sections(self):
        self._emiss_cross_sections = np.zeros((self.n_x_ray_transitions, self.n_epsilon))
        for i, xr in enumerate(self._x_ray_transitions):
            for j, e in enumerate(self.epsilons_J):
                self._emiss_cross_sections[i, j] = xr.emission_cross_section(e)

    @property
    def specific_stopping_power(self) -> np.ndarray:
        return self._specific_stopping_power

    @property
    def specific_stopping_power_d(self) -> np.ndarray:
        return self._specific_stopping_power_d

    @property
    def specific_transport_coefficient(self) -> np.ndarray:
        return self._specific_transport_coefficient

    def _update_specific_stopping_power_and_transport_coefficient(self):
        self._specific_stopping_power = np.zeros((self.n_elements, self.n_epsilon))
        self._specific_stopping_power_d = np.zeros((self.n_elements, self.n_epsilon))
        self._specific_transport_coefficient = np.zeros((self.n_elements, self.n_epsilon))
        for i, e in enumerate(self.elements):
            for j, epsilon in enumerate(self.epsilons_J):
                self._specific_stopping_power[i, j] = e.specific_stopping_power(epsilon) 
                self._specific_stopping_power_d[i, j] = e.specific_stopping_power_d(epsilon)
                self._specific_transport_coefficient[i, j] = e.specific_transport_coefficient(epsilon)

    @property
    def attenuation_integration_segments(self) -> np.ndarray:
        return self._attenuation_integration_segments

    def _update_attenumation_integration_segments(self):
        #TODO: detach material from detector
        self._attenuation_integration_segments = self.detector._line_segments()

    @property
    def standard_intensities(self) -> np.ndarray:
        return self._standart_intensities

    def _update_standard_intensities(self):
        self._standart_intensities = np.array([xr.std_intensity for xr in self.x_ray_transitions])

    @property
    def specific_n_of_atoms(self) -> np.ndarray:
        return self._specific_n_of_atoms

    def _update_specific_n_of_atoms(self):
        self._specific_n_of_atoms =  np.array([xr.element.atoms_per_cubic_meter() for xr in self.x_ray_transitions])
    
    def _update_extraction_operator(self):
        def _k_ratios(parameters, electron_fluence):
            mass_fractions = mass_fractions_from_parameters(self.material.n_x, self.material.n_y, parameters)
            return k_ratio_model.k_ratios(
                mass_fractions=mass_fractions,
                specific_densities=self.specific_densities,
                specific_n_of_atoms=self.specific_n_of_atoms,
                line_segments=self.attenuation_integration_segments,
                specific_attenuation_coefficients=self.specific_attenuation_coefficients,
                emiss_cross_sections=self.emiss_cross_sections,
                electron_fluence=electron_fluence,
                standart_intensities=self.standard_intensities,
                number_of_cells_per_subdomain=self.material.number_of_cells_per_subdomain,
                delta_epsilon=self.delta_epsilon_J,
                delta_x=self.material.delta_x,
                delta_y=self.material.delta_y)
        # self.extraction_operator = jax.jit(_k_ratios)
        self.extraction_operator = _k_ratios
        # self.extraction_operator_jacobian = jax.jit(jax.jacrev(_k_ratios, argnums=0))
        self.extraction_operator_jacobian = jax.jacrev(_k_ratios, argnums=0)
        def _scalar_product(parameters, solution_forward, solution_adjoint):
            mass_fractions = mass_fractions_from_parameters(self.material.n_x, self.material.n_y, parameters)
            return k_ratio_model.scalar_product(
                mass_fractions=mass_fractions,
                solution_forward=solution_forward,
                solution_adjoint=solution_adjoint,
                specific_densities=self.specific_densities,
                specific_stopping_power=self.specific_stopping_power,
                specific_stopping_power_d=self.specific_stopping_power_d,
                specific_transport_coefficient=self.specific_transport_coefficient,
                number_of_cells_per_subdomain=self.material.number_of_cells_per_subdomain,
                delta_epsilon=self.delta_epsilon_J,
                delta_x=self.material.delta_x,
                delta_y=self.material.delta_y)
        # self.scalar_product = jax.jit(_scalar_product)
        self.scalar_product = _scalar_product
        # self.scalar_product_jacobian = jax.jit(jax.jacrev(_scalar_product, argnums=0)) 
        self.scalar_product_jacobian = jax.jacrev(_scalar_product, argnums=0)
        

    @property
    def parameter_dimensions(self):
        return (self.material.n_x, self.material.n_y, self.n_elements - 1)

    def update_std_intensities(self, std_ints = None):
        if std_ints == None:
            for x_ray in self.x_ray_transitions:
                el_idx = self.elements.index(x_ray.element)
                parameters = np.zeros(self.parameter_dimensions)
                if el_idx != len(self.elements) - 1:
                    parameters[:, :, el_idx] = 1.
                k_r = k_ratios(self, parameters)
                self._standart_intensities[el_idx] = k_r[el_idx]
        else:
            for i, std_int in enumerate(std_ints):
                self._standart_intensities[i] = std_int


def mass_fractions_from_parameters(n_x: int, n_y: int, parameters: np.ndarray):
    return jnp.append(parameters, jnp.reshape(1.-jnp.sum(parameters, axis=2), (n_x, n_y, 1)), axis=2)

# def extraction_operator(
#     experiment: Experiment,
#     parameters: np.ndarray,
#     forward_data: Dict):
#     ex = experiment
#     mass_fractions = mass_fractions_from_parameters(ex, parameters)
#     k_r = k_ratio_model.k_ratios(
#         mass_fractions=mass_fractions,
#         specific_densities=ex.specific_densities,
#         specific_n_of_atoms=ex.specific_n_of_atoms,
#         line_segments=ex.attenuation_integration_segments,
#         specific_attenuation_coefficients=ex.specific_attenuation_coefficients,
#         emiss_cross_sections=ex.emiss_cross_sections,
#         electron_fluence=forward_data['solution'][:, 0, :, :],
#         standart_intensities=ex.standard_intensities,
#         number_of_cells_per_subdomain=ex.material.number_of_cells_per_subdomain,
#         delta_epsilon=ex.delta_epsilon_J,
#         delta_x=forward_data['delta_x'],
#         delta_y=forward_data['delta_y']
#     )
#     return k_r

# def scalar_product(experiment, parameters, forward_data, adjoint_data):
#     ex = experiment
#     mass_fractions = mass_fractions_from_parameters(experiment.material.n_x, experiment.material.n_y, parameters)
#     return k_ratio_model.scalar_product(
#         mass_fractions=mass_fractions,
#         solution_forward=forward_data['solution'],
#         solution_adjoint=adjoint_data['solution'],
#         specific_densities=ex.specific_densities,
#         specific_stopping_power=ex.specific_stopping_power,
#         specific_stopping_power_d=ex.specific_stopping_power_d,
#         specific_transport_coefficient=ex.specific_transport_coefficient,
#         number_of_cells_per_subdomain=ex.material.number_of_cells_per_subdomain,
#         delta_epsilon=ex.delta_epsilon_J,
#         delta_x=forward_data['delta_x'],
#         delta_y=forward_data['delta_y'])

def k_ratios(experiment: Experiment, parameters: np.ndarray):
    mass_fractions = mass_fractions_from_parameters(experiment.material.n_x, experiment.material.n_y, parameters)
    forward_data = m1model.solve_forward(experiment, mass_fractions)
    k_r = experiment.extraction_operator(parameters, forward_data['solution'][:, 0, :, :])
    del mass_fractions, forward_data
    return k_r

def k_ratios_jacobian(experiment: Experiment, parameters: np.ndarray):
    mass_fractions = mass_fractions_from_parameters(experiment.material.n_x, experiment.material.n_y, parameters)
    print("solving forward")
    forward_data = m1model.solve_forward(experiment, mass_fractions)
    print("extraction operator")
    k_r = experiment.extraction_operator(parameters, forward_data['solution'][:, 0, :, :])
    print("extraction opertator jacobian")
    k_r_jacobian = experiment.extraction_operator_jacobian(parameters, forward_data['solution'][:, 0, :, :])
    print("calculating")
    densities = k_ratio_model.densities(mass_fractions, experiment.specific_densities)
    att_coeff = k_ratio_model.attenuation_coefficients(mass_fractions, densities, experiment.attenuation_integration_segments, experiment.specific_attenuation_coefficients)
    n_of_atoms = k_ratio_model.number_of_atoms(mass_fractions, densities, experiment.specific_n_of_atoms, experiment.material.number_of_cells_per_subdomain)
    
    for k in range(experiment.n_x_ray_transitions):
        def adjoint_source(counter):
            step_count = counter.i
            return n_of_atoms[k, :, :] * att_coeff[:, :, k] * experiment.emiss_cross_sections[k,step_count] / experiment.standard_intensities[k]
        print("solving adjoint")
        adjoint_data = m1model_adjoint.solve_adjoint(experiment, adjoint_source, mass_fractions, forward_data['solution'])
        # temp = scalar_product(experiment, parameters, forward_data, adjoint_data)
        # scp_jacobian = - jacrev(lambda p: scalar_product(experiment, p, forward_data, adjoint_data))(parameters)
        print("scalar product jacobian")
        # temp = experiment.scalar_product(parameters, forward_data['solution'], adjoint_data['solution'])
        scp_jacobian = -experiment.scalar_product_jacobian(parameters, forward_data['solution'], adjoint_data['solution'])
        print("adding to jacobian")
        k_r_jacobian = jax.ops.index_add(k_r_jacobian, jax.ops.index[k, :, :, :], scp_jacobian)
    del mass_fractions, forward_data, densities, att_coeff, n_of_atoms, adjoint_data, scp_jacobian
    return k_r, k_r_jacobian


