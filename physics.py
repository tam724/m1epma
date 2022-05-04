from scipy.constants import mega, kilo, electron_volt, nano
import scipy.stats
import math
import numpy as np
import json
import os
from typing import List

keV = kilo*electron_volt

dirname = os.path.dirname(os.path.realpath(__file__))

ELEM_CHARGE = 1.6021766208e-19
VAC_PERMIT = 8.854187817e-12
PI = 3.14159265359
E = 2.7182818284590452353602874713527
B = math.sqrt(0.5 * E)
H = 6.62607004e-34
MASS_ELECTRON = 9.10938356e-31
AVOGADRO = 6.02214086e23
PLANCK_CONSTANT = 6.62606957e-34
BOHR_RADIUS = 5.2917721092e-11
RYDBERG_ENERGY = 13.60569253 * ELEM_CHARGE
ELECTRON_MASS_ENERGY_EQUIVALENT = 8.18710506e-14

FFastEdgeDB = np.loadtxt(
    dirname + '/data/FFastEdgeDB.csv', delimiter=',') * ELEM_CHARGE
GroundStateOccupancies = np.loadtxt(
    dirname + '/data/GroundStateOccupancies.csv', delimiter=',')
nSubShells = [1, 3, 5]
shellIdx = [[0, 0, 0, 0, 0], [1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]
fluorescence_coefficients = [[1.4340e-1, -2.5606e-2, 1.3163e-3, 0, 0],
                             [-7.6388e-1, 5.4070e-2, -4.0544e-4, -1.4348e-6, 1.8252e-8]]
mass_absorption_table = json.load(
    open(dirname + '/data/mass_absorption_coefficient.json'))


class Element(object):
    def __init__(self, name, atomic_number, molar_mass, density):
        self.z = atomic_number
        self.M = molar_mass
        self._rho = density
        self.name = name

    def density(self):
        return self._rho

    def atomic_number(self):
        return self.z

    def atomic_mass(self):
        return self.M / AVOGADRO

    def molar_mass(self):
        return self.M

    def mass_absorption_coefficient(self, x_ray_energy):
        z = self.atomic_number()
        return np.interp(x_ray_energy / (mega * ELEM_CHARGE), mass_absorption_table[str(z)]['energy'],
                         mass_absorption_table[str(z)]["my/rho"]) * .1  # cm**2/g to m**2/kg

    def specific_stopping_power(self, electron_energy):
        """ returns the specific stopping power for this element for an electron energy"""
        z = self.atomic_number()
        a = self.atomic_mass()
        eps = electron_energy
        if z > 6:
            j = ELEM_CHARGE * (9.76 * z + 58.8 * (z ** (-0.19)))
        else:
            j = ELEM_CHARGE * (11.5 * z)
        c1 = (1.0 / a) * ((2. * PI * (ELEM_CHARGE ** 4.) * z) /
                          ((4. * PI * VAC_PERMIT) ** 2.))
        # TODO: MAY BE AN ERROR IN Mevenkamp p. 14
        # C1 = (rho(i)/A(i))*((2.d0*PI*(ELEM_CHARGE**4.d0)*Z(i))/((2.d0*PI*VAC_PERMIT)**2.d0))
        c2 = math.log((B * eps) / j)
        s = c1 * c2 / eps
        return s

    def specific_stopping_power_d(self, electron_energy):
        """ returns the derivative of the specific stopping power w.r.t. the electron energy"""
        z = self.atomic_number()
        a = self.atomic_mass()
        eps = electron_energy
        if z > 6:
            j = ELEM_CHARGE * (9.76 * z + 58.8 * (z ** (-0.19)))
        else:
            j = ELEM_CHARGE * (11.5 * z)
        c1 = (1.0 / a) * ((2. * PI * (ELEM_CHARGE ** 4.) * z) /
                          ((4. * PI * VAC_PERMIT) ** 2.))
        # TODO: MAY BE AN ERROR IN Mevenkamp p. 14
        # C1 = (rho(i)/A(i))*((2.d0*PI*(ELEM_CHARGE**4.d0)*Z(i))/((2.d0*PI*VAC_PERMIT)**2.d0))
        c2 = math.log((B * eps) / j)
        # NOTE: d/dt S(x,eps(t)) = d/deps S(x,eps) d/dt eps(t) = - d/deps S(x,eps)
        ds = c1 * (1 - c2) / (eps ** 2.)
        return ds

    def specific_stopping_power_integral(self, electron_energy):
        # type: (float) -> float
        z = self.atomic_number()
        a = self.atomic_mass()
        eps = electron_energy
        if z > 6:
            j = ELEM_CHARGE * (9.76 * z + 58.8 * (z ** (-0.19)))
        else:
            j = ELEM_CHARGE * (11.5 * z)
        c1 = (1.0 / a) * ((2. * PI * (ELEM_CHARGE ** 4.) * z) /
                          ((4. * PI * VAC_PERMIT) ** 2.))
        c2 = math.log((B * eps) / j)
        s_integral = 0.5*c1*c2**2.
        return s_integral

    def specific_transport_coefficient(self, electron_energy):
        z = self.atomic_number()
        a = self.atomic_mass()
        eps = electron_energy
        # calculate screening angle
        theta = (H / math.sqrt(2. * MASS_ELECTRON * eps)) / (
            2. * PI * (H ** 2. * VAC_PERMIT) / (PI * MASS_ELECTRON * ELEM_CHARGE ** 2.) * (z ** (-1. / 3.)))
        tc = (2. * PI * ELEM_CHARGE ** 4.) / (16. * (4. * PI * VAC_PERMIT) ** 2. * eps ** 2.) * (z ** 2.) / a * (
                8. / (math.cos(theta) - 3.) + 4. * (math.log(3. - math.cos(theta)) - math.log(1. - math.cos(theta))))
        return tc

    def atoms_per_cubic_meter(self):
        return AVOGADRO / self.molar_mass()

    def fluorescence_yield_k_shell(self):
        w_k = 0.0
        z = self.atomic_number()
        if z < 11:
            raise ValueError
        elif z < 20:
            coeff = fluorescence_coefficients[0]
        elif z < 99:
            coeff = fluorescence_coefficients[1]
        else:
            raise ValueError

        for n in range(0, len(coeff)):
            w_k += coeff[n] * z ** n
        return w_k


class Copper(Element):
    def __init__(self):
        super(Copper, self).__init__(name="Copper",
                                     atomic_number=29,
                                     molar_mass=63.546e-3,
                                     density=8.96e3)


class Nickel(Element):
    def __init__(self):
        super(Nickel, self).__init__(name="Nickel",
                                     atomic_number=28,
                                     molar_mass=58.6934e-3,
                                     density=8.902e3)


class Chromium(Element):
    def __init__(self):
        super(Chromium, self).__init__(name="Chromium",
                                       atomic_number=24,
                                       molar_mass=51.9961e-3,
                                       density=7.19e3)


class Silicon(Element):
    def __init__(self):
        super(Silicon, self).__init__(name="Silicon",
                                      atomic_number=14,
                                      molar_mass=28.0855e-3,
                                      density=2.329e3)


class Iron(Element):
    def __init__(self):
        super(Iron, self).__init__(name="Iron",
                                   atomic_number=26,
                                   molar_mass=55.845e-3,
                                   density=7.874e3)


class Gold(Element):
    def __init__(self):
        super(Gold, self).__init__(name="Gold",
                                   atomic_number=79,
                                   molar_mass=196.96657e-3,
                                   density=19.30e3)

class Manganese(Element):
    def __init__(self):
        super(Manganese, self).__init__(name="Manganese",
                                        atomic_number=25,
                                        molar_mass=54.938044e-3,
                                        density=7.44e3)

class Aluminium(Element):
    def __init__(self):
        super(Aluminium, self).__init__(name="Aluminium",
                                        atomic_number=13,
                                        molar_mass=26.98154e-3, 
                                        density=2.6989e3)

class Gallium(Element):
    def __init__(self):
        super(Gallium, self).__init__(name="Gallium",
                                      atomic_number=31,
                                      molar_mass=69.72e-3,
                                      density=5.91e3)

class Arsenic(Element):
    def __init__(self):
        super(Arsenic, self).__init__(name="Arsenic",
                                      atomic_number=33,
                                      molar_mass=74.9216e-3,
                                      density=5.727e3)

class Carbon(Element):
    def __init__(self):
        super(Carbon, self).__init__(name="Carbon",
                                      atomic_number=6,
                                      molar_mass=12.011e-3,
                                      density=2.260e3)

class Titan(Element):
    def __init__(self):
        super(Titan, self).__init__(name="Titan",
                                    atomic_number=22,
                                    molar_mass=47.867e-3,
                                    density=4.54e3)


class XRay(object):
    def __init__(self, element, std_intensity):
        # currently only supporting one characteristic x-ray per element
        self.element = element # the element the x-ray is characteristic to
        self.type = "K-alpha" # the type of the x-ray
        self.std_intensity = std_intensity # the standard intensity of the x-ray

    def get_std_intensity(self):
        return self.std_intensity

    def photon_energy(self):
        """ using the mean k-alpha (1 and 2) photon energy here"""
        edge_energies = FFastEdgeDB[self.element.atomic_number() - 1]
        e1 = edge_energies[shellIdx[0][0]]
        energy = 0.
        for j in range(1, 3):
            energy += e1 - edge_energies[shellIdx[1][j]]
        return energy / 2.

    def family(self):
        return 0

    def ionization_cross_section(self, energy):
        res = 0.
        for subShell in range(0, nSubShells[self.family()]):
            ground_state_occupancy = get_ground_state_occupancy(
                self.element.atomic_number(), self.family(), subShell)
            ee = get_edge_energy(self.element.atomic_number(),
                                 self.family(), subShell)
            u = energy / ee
            if u > 1.0:
                phi = 10.57 * math.exp(-1.736 / u + 0.317 / u ** 2.)
                psi = math.pow(ee / RYDBERG_ENERGY, -0.0318 +
                               (0.3160 / u) + (-0.1135 / u ** 2))
                i = ee / ELECTRON_MASS_ENERGY_EQUIVALENT
                t = energy / ELECTRON_MASS_ENERGY_EQUIVALENT
                f = ((2.0 + i) / (2.0 + t)) * math.pow((1.0 + t) / (1.0 + i), 2) * math.pow(
                    ((i + t) * (2.0 + t) * math.pow(1.0 + i, 2)) / (
                                t * (2.0 + t) * math.pow(1.0 + i, 2) + i * (2.0 + i)), 1.5)
                res += (ground_state_occupancy * math.pow(BOHR_RADIUS *
                                                          RYDBERG_ENERGY / ee, 2) * f * psi * phi * math.log(u) / u)
        return res

    def emission_cross_section(self, energy):
        return self.ionization_cross_section(energy)*self.element.fluorescence_yield_k_shell()

def normalize(x):
    return x / np.linalg.norm(x)


class ElectronBeam:
    def __init__(self, size, pos, beam_energy_keV, energy_variation_keV):
        self.size_x = size[0]
        self.size_y = size[1]
        self.position_x = pos[0]
        self.position_y = pos[1]
        self.energy_variation_keV = energy_variation_keV
        self.beam_energy_keV = beam_energy_keV

    def intensity_dist(self, x, y, eps_keV):
        return scipy.stats.multivariate_normal(mean=self.position_x, cov=self.size_x).pdf(x)*\
            scipy.stats.multivariate_normal(mean=self.beam_energy_keV, cov=self.energy_variation_keV).pdf(eps_keV)
        # return scipy.stats.multivariate_normal(mean=[self.position_x, self.position_y], cov=np.diag([self.size_x, self.size_y])).pdf([x, y])*\
        #     scipy.stats.multivariate_normal(mean=self.beam_energy_keV, cov=self.energy_variation_keV).pdf(eps_keV)
        # return np.exp(-np.power(1 / self.energy_variation * (eps - self.beam_energy), 2))

class Material:
    def __init__(self, n_x, n_y, hat_n_x, hat_n_y, dim_x, dim_y):
        self._n_x = n_x # reconstruction pixel grid
        self._n_y = n_y # reconstruction pixel grid
        self._hat_n_x = hat_n_x # clawpack FV grid
        self._hat_n_y = hat_n_y # clawpack FV grid
        self._dim_x = dim_x
        self._dim_y = dim_y
        self._update_number_of_cells_per_subdomain()
        self._update_x_y()

    @property
    def n_x(self):
        return self._n_x
    
    @property
    def n_y(self):
        return self._n_y
    
    @property
    def hat_n_x(self):
        return self._hat_n_x

    @property
    def hat_n_y(self):
        return self._hat_n_y

    @property
    def dim_x(self):
        return self._dim_x

    @property
    def dim_y(self):
        return self._dim_y

    @property
    def len_x(self):
        return self._dim_x[1] - self._dim_x[0]
    
    @property
    def len_y(self):
        return self._dim_y[1] - self._dim_y[0]

    @property
    def slices_x(self):
        return np.linspace(self._dim_x[0], self._dim_x[1], self._n_x + 1)
    
    @property
    def slices_y(self):
        return np.linspace(self._dim_y[0], self._dim_y[1], self._n_y + 1)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def delta_x(self):
        return (self.dim_x[1] - self.dim_x[0])/self.hat_n_x

    @property
    def delta_y(self):
        return (self.dim_y[1] - self.dim_y[0])/self.hat_n_y

    def _update_x_y(self):
        x = np.linspace(self.dim_x[0] + self.delta_x/2., self.dim_x[1] - self.delta_x/2., self.hat_n_x)
        y = np.linspace(self.dim_y[0] + self.delta_y/2., self.dim_y[1] - self.delta_y/2., self.hat_n_y)
        self._y, self._x = np.meshgrid(y, x)

    @property
    def number_of_cells_per_subdomain(self):
        return self._number_of_cells_per_subdomain

    def _update_number_of_cells_per_subdomain(self):
        self._number_of_cells_per_subdomain =  (self.hat_n_x//self.n_x, self.hat_n_y//self.n_y)
    

def map_from_to(x, from_low, from_high, to_low, to_high):
    y = (x-from_low)/(from_high-from_low)*(to_high-to_low)+to_low
    return y

class Detector(object):
    def __init__(self, x:float, y:float, material:Material):
        self.x = x  # x position of the detector
        self.y = y  # y position of the detector
        self.material = material
        # self.l_segments = self._line_segments()

    def _line_segments_through_material(self, x_source, y_source):
        """
        :param x_source: x position of the source
        :param y_source: y position of the source
        :param x_ray: x_ray
        :return: np.array(mx, my) (material dimensions) containing the length of the ray through each cell
        """
        points = [{'m': 0., 'x': x_source, 'y': y_source}]  # assuming x_source and y_source is inside the material
        if self.material.dim_x[0] <= self.x <= self.material.dim_x[1] and self.material.dim_y[0] <= self.y <= self.material.dim_y[1]:
            points.append({'m': 1., 'x': self.x, 'y': self.y})
        if self.x - x_source != 0:
            for x in self.material.slices_x:
                m = (x - x_source) / (self.x - x_source)
                y = y_source + m * (self.y - y_source)
                if 0. < m < 1. and self.material.dim_y[0] <= y <= self.material.dim_y[1]:
                    points.append({'m': m, 'x': x, 'y': y})
        if self.y - y_source != 0:
            for y in self.material.slices_y:
                m = (y - y_source)/(self.y - y_source)
                x = x_source + m * (self.x - x_source)
                if 0. < m < 1. and self.material.dim_x[0] <= x <= self.material.dim_x[1]:
                    if not any([math.isclose(p_['m'], m) for p_ in points]):
                        points.append({'m': m, 'x': x, 'y': y})
        points = sorted(points, key=lambda p: p['m'])
        length = np.sqrt(np.power(self.x - x_source, 2.) + np.power(self.y - y_source, 2.))
        segments = np.zeros((self.material.n_x, self.material.n_y))
        for i in range(len(points) - 1):
            x_m = (points[i]['x'] + points[i+1]['x'])/2.
            y_m = (points[i]['y'] + points[i+1]['y'])/2.
            is_ = int(map_from_to(x_m, self.material.dim_x[0], self.material.dim_x[1], 0., self.material.n_x))
            js_ = int(map_from_to(y_m, self.material.dim_y[0], self.material.dim_y[1], 0., self.material.n_y))
            ## HACK: DO NOT THINK ABOUT THIS!
            if is_ == self.material.n_x-1:
                ## COMPUTE THE INTERSECTION WITH THE MATERIAL SURFACE
                m = (0. - y_source)/(self.y - y_source)
                segments[is_, js_] += (m - points[i]['m'])*length
            else:
                # ONLY THIS EXCLUDES REGIONS OUTSIDE OF THE COMPUTATIONAL DOMAIN
                segments[is_, js_] += (points[i+1]['m'] - points[i]['m'])*length
        return points, segments

    def _line_segments(self):
        mx, my = self.material.hat_n_x, self.material.hat_n_y
        m_x, m_y = self.material.n_x, self.material.n_y
        # return np.zeros((mx, my, m_x, m_y))
        line_segments = np.zeros((mx, my, m_x, m_y))
        for i in range(mx):
            for j in range(my):
                _, sgs = self._line_segments_through_material(self.material.x[i, j], self.material.y[i, j])
                line_segments[i, j, :, :] = sgs
        return line_segments

    # def _attenuation_coefficients(self, material):
    #     mx, my = self.m_geometry.mx, self.m_geometry.my
    #     coeffs = np.zeros((len(self.x_rays), mx, my))
    #     # coeffs2 = np.zeros((len(self.x_rays), mx, my))
    #     for k in range(len(self.x_rays)):
    #         x_ray = self.x_rays[k]
    #         atten_coeff_cellwise = material.linear_attenuation_coefficient_cellwise(x_ray.photon_energy())
    #         # for i in range(mx):
    #         #     for j in range(my):
    #         #         for l in range(self.m_geometry.m_x):
    #         #             for m in range(self.m_geometry.m_y):
    #         #                 coeffs2[k, i, j] += atten_coeff_cellwise[l, m]*self.l_segments[i, j, l, m]
    #         #         coeffs2[k, i, j] = np.exp(-coeffs2[k, i, j])
    #         coeffs[k, :, :] = np.exp(-np.tensordot(atten_coeff_cellwise, self.l_segments, axes=[(0, 1), (2, 3)]))
    #     return coeffs

    # def _attenuation_coefficients_grad(self, material):
    #     mx, my = self.m_geometry.mx, self.m_geometry.my
    #     px, py, pe = material.parameters.shape
    #     kk = len(self.x_rays)
    #     coeffs_grad = np.zeros((kk, mx, my, px, py, pe))
    #     coeffs = self._attenuation_coefficients(material)
    #     for k in range(kk):
    #         spec_atten_coeff = [e.mass_absorption_coefficient(self.x_rays[k].photon_energy()) for e in material.elements]
    #         atten_coeff_cellwise_gradient = material._material_property_cellwise_gradient(spec_atten_coeff)
    #         coeffs_grad = -np.einsum('kij,ijlm,lmn->kijlmn', coeffs, self.l_segments, atten_coeff_cellwise_gradient, optimize=True)
    #     #     for i in range(mx):
    #     #         for j in range(my):
    #     #             for l in range(material.m_x):
    #     #                 for m in range(material.m_y):
    #     #                     coeffs_grad[k, i, j, l, m, :] += -coeffs[k, i, j]*line_segments[i, j, l, m]*atten_coeff_cellwise_gradient[l, m, :]
    #     return coeffs_grad

    # def x_ray_intensity(self, solution, material):
    #     x_ray_int = np.zeros((len(self.x_rays), ) + solution.x.shape)
    #     N = len(solution.t)
    #     deps = solution.dt
    #     geom = solution.simulation_parameters.material.geometry
    #     # using the trapezoidal rule in time
    #     atten_coeffs = self._attenuation_coefficients(material)
    #     for k in range(0, len(self.x_rays)):
    #         x_ray = self.x_rays[k]
    #         I0 = np.zeros((geom.mx, geom.my))
    #         for i in range(0, N):
    #             eps = solution.energy_at(i)
    #             psi0 = solution.at_energy(i)[0, :, :]
    #             if i == 0 or i == N - 1:
    #                 trap_rl_fac = 0.5
    #             else:
    #                 trap_rl_fac = 1.
    #             sigma_emiss = x_ray.emission_cross_section(eps)
    #             I0 += trap_rl_fac * sigma_emiss * psi0 * deps
    #         N_V = material.expand_cellwise_property(material.number_of_atoms_cellwise(x_ray.element),
    #                                                 [geom.mx, geom.my])
    #         x_ray_int[k, :, :] = I0 * N_V * atten_coeffs[k, :, :]
    #     return x_ray_int

    # def measure(self, solution, material):
    #     k_ratios = np.zeros(len(self.x_rays))
    #     N = len(solution.t)
    #     deps = solution.dt
    #     geom = solution.simulation_parameters.material.geometry
    #     # using the trapezoidal rule in time
    #     atten_coeffs = self._attenuation_coefficients(material)
    #     for k in range(0, len(self.x_rays)):
    #         x_ray = self.x_rays[k]
    #         I0 = np.zeros((geom.mx, geom.my))
    #         for i in range(0, N):
    #             eps = solution.energy_at(i)*keV
    #             psi0 = solution.at_energy(i)[0, :, :]
    #             if i == 0 or i == N - 1:
    #                 trap_rl_fac = 0.5
    #             else:
    #                 trap_rl_fac = 1.
    #             sigma_emiss = x_ray.emission_cross_section(eps)
    #             I0 += trap_rl_fac*sigma_emiss*psi0*deps
    #         N_V = material.expand_cellwise_property(material.number_of_atoms_cellwise(x_ray.element), [geom.mx, geom.my])
    #         k_ratios[k] = np.sum(I0*N_V*atten_coeffs[k, :, :])*solution.dx*solution.dy/x_ray.get_std_intensity()
    #     return k_ratios

    # def calculate_partial_gradient(self, U, material, measurement, ref_measurement, adj_sim_params):
    #     deps = U.dt
    #     dx = U.dx
    #     dy = U.dy
    #     N = len(U.t)
    #     geom = material.geometry

    #     atten_coeffs_grad = self._attenuation_coefficients_grad(material)
    #     atten_coeffs = self._attenuation_coefficients(material)
    #     partial_gradient = np.zeros(material.parameters.shape)
    #     if adj_sim_params.k is None:
    #         difference = measurement - ref_measurement
    #         for k in range(0, len(self.x_rays)):
    #             x_ray = self.x_rays[k]
    #             I0 = np.zeros((geom.mx, geom.my))
    #             for i in range(0, N):
    #                 eps = U.energy_at(i)
    #                 psi0 = U.at_energy(i)[0, :, :]
    #                 if i == 0 or i == N - 1:
    #                     trap_rl_fac = 0.5
    #                 else:
    #                     trap_rl_fac = 1.
    #                 sigma_emiss = x_ray.emission_cross_section(eps)
    #                 I0 += trap_rl_fac * sigma_emiss * psi0 * deps
    #             N_V = material.expand_cellwise_property(material.number_of_atoms_cellwise(x_ray.element),
    #                                                     [geom.mx, geom.my])
    #             N_V_grad_cellw = material.number_of_atoms_cellwise_gradient(x_ray.element)
    #             partial_gradient += difference[k] * np.tensordot(I0 * N_V, atten_coeffs_grad[k],
    #                                                      axes=[(0, 1), (0, 1)]) * dx * dy / x_ray.get_std_intensity()

    #             n_cells = U.x.shape  # m1model cells
    #             n_per_cell_x = n_cells[0] / geom.m_x  # number of m1model cells per material cell
    #             n_per_cell_y = n_cells[1] / geom.m_y
    #             for i in range(0, geom.m_x):
    #                 for j in range(0, geom.m_y):
    #                     indices = (slice(i * n_per_cell_x, (i + 1) * n_per_cell_x),
    #                                slice(j * n_per_cell_y, (j + 1) * n_per_cell_y))  # indices of this material cell
    #                     a_coeffs = atten_coeffs[k]
    #                     partial_gradient[i, j, :] += difference[k] * np.sum(I0[indices] * a_coeffs[indices]) * N_V_grad_cellw[
    #                         i, j] * dx * dy / x_ray.get_std_intensity()
    #     else:
    #         k = adj_sim_params.k
    #         x_ray = self.x_rays[k]
    #         I0 = np.zeros((geom.mx, geom.my))
    #         for i in range(0, N):
    #             eps = U.energy_at(i)
    #             psi0 = U.at_energy(i)[0, :, :]
    #             if i == 0 or i == N - 1:
    #                 trap_rl_fac = 0.5
    #             else:
    #                 trap_rl_fac = 1.
    #             sigma_emiss = x_ray.emission_cross_section(eps)
    #             I0 += trap_rl_fac * sigma_emiss * psi0 * deps
    #         N_V = material.expand_cellwise_property(material.number_of_atoms_cellwise(x_ray.element),
    #                                                 [geom.mx, geom.my])
    #         N_V_grad_cellw = material.number_of_atoms_cellwise_gradient(x_ray.element)
    #         partial_gradient += np.tensordot(I0 * N_V, atten_coeffs_grad[k],
    #                                                          axes=[(0, 1),
    #                                                                (0, 1)]) * dx * dy / x_ray.get_std_intensity()

    #         n_cells = U.x.shape  # m1model cells
    #         n_per_cell_x = n_cells[0] // geom.m_x  # number of m1model cells per material cell
    #         n_per_cell_y = n_cells[1] // geom.m_y
    #         for i in range(0, geom.m_x):
    #             for j in range(0, geom.m_y):
    #                 indices = (slice(i * n_per_cell_x, (i + 1) * n_per_cell_x),
    #                            slice(j * n_per_cell_y, (j + 1) * n_per_cell_y))  # indices of this material cell
    #                 a_coeffs = atten_coeffs[k]
    #                 partial_gradient[i, j, :] += np.sum(I0[indices] * a_coeffs[indices]) * \
    #                                              N_V_grad_cellw[
    #                                                  i, j] * dx * dy / x_ray.get_std_intensity()
    #     return partial_gradient

    # def calculate_gradient(self, U, ld, material, measurement, ref_measurement, adj_sim_params):
    #     partial_gradient = self.calculate_partial_gradient(U, material, measurement, ref_measurement, adj_sim_params)
    #     scalar_product = material.calculate_scalar_product(U, ld)
    #     return partial_gradient - scalar_product

    # def adjoint_source(self, eps, sim_params):
    #     solution = sim_params.forward_solution
    #     measurement = sim_params.measurement
    #     ref_measurement = sim_params.reference_measurement
    #     material = sim_params.material
    #     geom = material.geometry

    #     adjoint_src = np.zeros(solution.x.shape)
    #     difference = measurement - ref_measurement
    #     atten_coeffs = sim_params.get_from_cache('attenuation_coeffs')
    #     if sim_params.k is None:
    #         for k in range(0, len(self.x_rays)):
    #             x_ray = self.x_rays[k]
    #             sigma_emiss = x_ray.emission_cross_section(eps)
    #             N_V = material.expand_cellwise_property(material.number_of_atoms_cellwise(x_ray.element), [geom.mx, geom.my])
    #             adjoint_src += difference[k]*atten_coeffs[k, :, :]*sigma_emiss*N_V/x_ray.get_std_intensity()
    #     else:
    #         # calculate single derivative
    #         x_ray = self.x_rays[sim_params.k]
    #         sigma_emiss = x_ray.emission_cross_section(eps)
    #         N_V = material.expand_cellwise_property(material.number_of_atoms_cellwise(x_ray.element), [geom.mx, geom.my])
    #         adjoint_src += atten_coeffs[sim_params.k, :, :]*sigma_emiss*N_V/x_ray.get_std_intensity()
    #     return adjoint_src

    # def cost_functional(self, for_meas, ref_meas):
    #     return 1. / 2. * np.sum(np.power(for_meas - ref_meas, 2.))



    # def plot_energy_scanning_rate(self, eps):
    #     eps_initial = max(eps)
    #     eps_cutoff = min(eps)
    #     ene = np.zeros(eps.shape)
    #     eps_plot = np.linspace(eps_cutoff, eps_initial, len(eps)*3)
    #     ene_plot = np.zeros(eps_plot.shape)
    #     for i in range(0, len(eps)*3):
    #         ene_plot[i] = self.intensity_dist(eps_plot[i])
    #     for i in range(0, len(eps)):
    #         ene[i] = self.intensity_dist(eps[i])
    #     plt.plot(eps_plot, ene_plot, label="intensity")
    #     plt.plot(eps, ene, 'o', label="scanning")
    #     plt.legend()
    #     plt.show()



def get_edge_energy(z, x_ray_family, sub_shell):
    return FFastEdgeDB[z - 1][shellIdx[x_ray_family][sub_shell]]


def get_ground_state_occupancy(z, x_ray_family, sub_shell):
    return GroundStateOccupancies[z - 1][shellIdx[x_ray_family][sub_shell]]