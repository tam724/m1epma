# Attenuation Coefficients

## Exponential Attenuation Law [*[NIST]*](##References)
$$ I = I_0 \exp{(-\left(\frac{\mu}{\rho}\right) \rho x)}$$
where
$(\frac{\mu}{\rho})$ is the mass attenuation coefficient,
$x$ the depth.
The mass attenuation coefficient for a compound is
$$ \left(\frac{\mu}{\rho}\right)_k = \sum_{e=1}^{n_e} c_e \bar{\left(\frac{\mu}{\rho}\right)}_{ke}$$
where
$c_e$ is the weight fraction of element $e$.
The mass attenuation coefficient does depend on the wavelength/energy of the absorbed photon (therefore the $\cdot_k$).

If the density or the attenuation coefficient if not constant over the depth, the attenuation factor of can be written as
$$ A_k(x) = \exp{(-\int_{d(x)} \mu_k(y) dy})$$

Analogous to the (generic, multidimensional) material property ($\mu \in \mathbb{R}^{n_k \times n_x \times n_y}$}):
$$\mu_{kpq} = \sum_{e=1}^{n_e} c_{pqe}\rho_{pq}\bar{\left(\frac{\mu}{\rho}\right)}_{ke}$$

Using the line segments [*[ref]*](##Line-Segments) $L \in \mathbb{R}^{\hat{n}_x \times \hat{n}_y \times n_x \times n_y}$ $A$ simplifies to
$$ A_{ijk} = \exp{(-\sum_{p=1}^{n_x} \sum_{q=1}^{n_y} L_{ijpq} \mu_{kpq})}$$ 

## Derivative of the Attenuation
$$\frac{\partial A_{ijk}}{\partial c_{mnq}} $$
**[TODO]** add the derivative

## Line Segments
**[TODO]** add line segments description

## References
- [NIST] <https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients>
