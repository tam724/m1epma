import jax.numpy as np
import jax as jax
from typing import List, Tuple
jax.config.update("jax_enable_x64", True)

def densities(
    mass_fractions: np.ndarray,
    specific_densities: np.ndarray) -> np.ndarray:
    """
    Approximates the density of a compound.

    Parameters
    ----------
    mass_fractions: array_like
        Mass Fractions. Dimensions (n_x, n_y, n_e)
    specific_densities: array_like
        Densities of pure materials. Dimensions: (n_e)
    """
    return 1./np.einsum("ijk,k->ij", mass_fractions, 1./specific_densities)

def weighted_material_property(
    mass_fractions: np.ndarray,
    densities: np.ndarray,
    specific_material_property: np.ndarray) -> np.ndarray:
    """
    Calculates a (generic) material property on each of the material subdomains.
    The material property should be a 2D np.ndarray.
    (dimension of the material property: n_p, number of elements: n_e)

    Parameters
    ----------
    mass_fractions : array_like
        Mass fractions. Dimensions: (n_x, n_y, n_e)
    densities : array_like
        Densities. Dimensions: (n_x, n_y)
    specific_material_property : array_like
        Specific material property. Dimensions: (n_p, n_e)

    Returns
    -------
    np.ndarray
        Weighted material property. Dimensions: (n_p, n_x, n_y)
    """
    return np.einsum('ije,ij,pe->pij',
                     mass_fractions,
                     densities,
                     specific_material_property)

def attenuation_coefficients(
    mass_fractions: np.ndarray,
    densities: np.ndarray,
    line_segments: np.ndarray,
    specific_attenuation_coefficients: np.ndarray) -> np.ndarray:
    """
    Calculates the attenuation coefficients A_ijk for each of the finite volume cells (i, j)
    
    Parameters
    ----------
    mass_fractions : array_like
        Mass Fractions. Dimensions: (n_y, n_y, n_e)
    densities : array_like
        Densities. Dimensions: (n_x, n_y)
    line_segments : array_like
        Line segments. Dimensions: (\\hat{n}_x, \\hat_{n}_y, n_x, n_y)
    specific_attenuation_coefficients : array_like
        Specific attenuation coefficients. Dimensions: (n_e, n_k)
    
    Returns
    -------
    np.ndarray
        Attenuation coefficients. Dimensions: (\\hat{n}_x, \\hat{n}_y, n_k)
    """
    mu = weighted_material_property(mass_fractions, densities, specific_attenuation_coefficients)
    return np.exp(-np.einsum('ijpq,kpq->ijk', line_segments, mu))

def number_of_atoms(
    mass_fractions: np.ndarray,
    densities: np.ndarray,
    specific_n_of_atoms: np.ndarray,
    number_of_cells_per_subdomain: Tuple[int, int]) -> np.ndarray:
    """
    Calculates the number of atoms per cubic meter.

    Parameters
    ----------
    mass_fracions : array_like
        Mass Fractions. Dimensions: (n_x, n_y, n_e)
    densities : array_like
        Densities. Dimensions: (n_x, n_y)
    specific_n_of_atoms : array_like
        Atoms per cubic meter. Note: for each k ratio. Dimensions (n_k)
    number_of_cells_per_subdomain : tuple of int
        Tuple describing the number of cells of the finite volume grid per material subdomain.
    
    Returns
    -------
    array_like
        Number of Atoms. Dimensions (n_k, \\hat{n}_x, \\hat{n}_y)
    """
    # BUG: here we assume that only one k-ratio per element is calculated
    noa = np.einsum("ije,ij,e->eij", mass_fractions, densities, specific_n_of_atoms)
    ext = np.ones((1, ) + number_of_cells_per_subdomain)
    return np.kron(noa, ext)
    
def k_ratios(
    mass_fractions: np.ndarray,
    specific_densities: np.ndarray,
    specific_n_of_atoms: np.ndarray,
    line_segments: np.ndarray,
    specific_attenuation_coefficients: np.ndarray,
    emiss_cross_sections: np.ndarray,
    electron_fluence: np.ndarray,
    standart_intensities: np.ndarray,
    number_of_cells_per_subdomain: Tuple[int, int],
    delta_epsilon: float,
    delta_x: float,
    delta_y: float) -> np.ndarray:
    """
    Calculates the k-ratios for each of the specified x_rays (n_k).
    The trapezoidal rule is applied for the integration w.r.t \\epsilon.

    Parameters
    ----------
    standart_intensities : array_like
        Standart Intensities. Dimensions: (n_k)
    number_of_atoms : array_like
        Number of atoms per unit volume (expanded to finite volume cells and k-ratios). Dimensions: (n_k, \\hat{n}_x, \\hat{n}_y)
    attenuation_coefficients : array_like
        Attenuation coefficients. Dimensions: (\\hat{n}_x, \\hat{n}_y, n_k)
    emiss_cross_sections : array_like
        Emission cross-section. Dimensions: (n_k, n_{\\epsilon})
    electron_fluence: array_like
        Electron fluence. Dimensions: (n_{\\epsilon}, \\hat{n}_x, \\hat{n}_y)
    delta_epsilon : float
        Energy discretization stepsize.
    delta_x : float
        Spatial discretization stepsize. (x-dimension)
    delta_y : float
        Spatial discretization stepsize. (y-dimension)

    Returns
    -------
    array_like
        k ratios. Dimension: (n_k)
    """
    _densities = densities(mass_fractions, specific_densities)
    _number_of_atoms = number_of_atoms(mass_fractions, _densities, specific_n_of_atoms, number_of_cells_per_subdomain)
    _attenuation_coefficients = attenuation_coefficients(mass_fractions, _densities, line_segments, specific_attenuation_coefficients)
    # trapezoidal integration rule:
    n_epsilon = electron_fluence.shape[0]
    temp = np.ones(n_epsilon)
    tempt = jax.ops.index_update(temp, 0, 0.5)
    trapez_int_rule = jax.ops.index_update(tempt, -1, 0.5)
    intensity = np.einsum('kij,ijk,kt,tij,t->k', _number_of_atoms, _attenuation_coefficients, emiss_cross_sections, electron_fluence, trapez_int_rule)
    return intensity*delta_epsilon*delta_x*delta_y/standart_intensities

def scalar_product(
    mass_fractions: np.ndarray,
    solution_forward: np.ndarray,
    solution_adjoint: np.ndarray,
    specific_densities: np.ndarray,
    specific_stopping_power: np.ndarray,
    specific_stopping_power_d: np.ndarray,
    specific_transport_coefficient: np.ndarray,
    number_of_cells_per_subdomain: Tuple[int, int],
    delta_epsilon: float,
    delta_x: float,
    delta_y: float) -> np.float64:
    """
    Calculates the scalar product.

    Parameters
    ----------
    mass_fractions : array_like
        Mass Fractions. Dimensions: (n_x, n_y, n_e)
    solution_forward : array_like
        Solution of the forward equation. Dimensions: (n_{\\epsilon}, 3, \\hat{n}_x, \\hat{n}_y)
    solution_adjoint : array_like
        Solution of the adjoint equation. Dimensions: (n_{\\epsilon}, 3, \\hat{n}_x, \\hat{n}_y)
    specific_densities: array_like
        Densities of pure materials. Dimensions: (n_e)
    specific_stopping_power : array_like
        Stopping power. Dimensions: (n_e, n_{\\epsilon})
    specific_stopping_power_d : array_like
        Derivative of the stopping power. Dimensions: (n_e, n_{\\epsilon})
    specific_transport_coefficient : array_like
        Transport Coefficient. Dimensions: (n_e, n_{\\epsilon})
    number_of_cells_per_subdomain : tuple of int
        Tuple describing the number of cells of the finite volume grid per material subdomain.
    delta_epsilon : float
        Energy discretization stepsize.
    delta_x : float
        Spatial discretization stepsize. (x-dimension)
    delta_y : float
        Spatial discretization stepsize. (y-dimension)
    """
    n_epsilon = solution_forward.shape[0]
    ext = np.ones(number_of_cells_per_subdomain + (1,), dtype=np.float64)
    temp = np.ones(n_epsilon, dtype=np.float64)*delta_epsilon
    tempt = jax.ops.index_update(temp, 0, 0.5*delta_epsilon)
    trapez_int_rule = jax.ops.index_update(tempt, -1, 0.5*delta_epsilon)

    mass_fractions_ext = np.kron(mass_fractions, ext)
    dens = densities(mass_fractions_ext, specific_densities)
    stopping_power = np.einsum('ije,ij,et->ijt', mass_fractions_ext, dens, specific_stopping_power)
    stopping_power_d = np.einsum('ije,ij,et->ijt', mass_fractions_ext, dens, specific_stopping_power_d)
    transport_coefficient = np.einsum('ije,ij,et->ijt', mass_fractions_ext, dens, specific_transport_coefficient)
    scalar_product = -np.einsum("tkij,ijt,tkij,t->", solution_adjoint[:, :, :, :]*delta_x*delta_y, stopping_power_d, solution_forward[:, :, :, :], trapez_int_rule)
    scalar_product += np.einsum("tkij,ijt,tkij,t->", solution_adjoint[:, 1:3, :, :]*delta_x*delta_y, transport_coefficient, solution_forward[:, 1:3, :, :], trapez_int_rule)
    # energy loop
    for i in range(0, n_epsilon):
        if i == 0:
            solution_forward_d = (solution_forward[i + 1, :, :, :] - solution_forward[i, :, :, :])/delta_epsilon
        elif i == n_epsilon-1:
            solution_forward_d = (solution_forward[i, :, :, :] - solution_forward[i - 1, :, :, :])/delta_epsilon
        else:
            solution_forward_d = (solution_forward[i + 1, :, :, :] - solution_forward[i - 1, :, :, :])/(2.*delta_epsilon)
        scalar_product += -trapez_int_rule[i]*np.einsum("kij,ij,kij->", solution_adjoint[i, :, :, :]*delta_x*delta_y, stopping_power[:,:,i], solution_forward_d[:, :, :])
    return scalar_product