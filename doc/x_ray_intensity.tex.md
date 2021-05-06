# X Ray Intensity / K-ratios
We calculate k-ratios $k \in \mathbb{R}^k$ (here $e$ is the index of the element $k$ is characteristic to)
$$ k_k = \frac{1}{I^{std}_k} \int_{\Omega} N^V_e(x) A_k(x) \int_{\epsilon}\sigma_k^{emiss} (\epsilon) \psi^0(x, \epsilon)d\epsilon dx$$

## Number of Atoms per Unit Volume
The number of atoms of element $e$ per unit volume is
$$N^V_e(x) = \rho_e(x) \frac{\mathcal{N}_A}{A_e}$$
where
$\mathcal{N}_A$ is avogardos number,
$A_e$ is the molar mass and 
$\rho_e(x) = c_e(x)\rho(x)$ is the partial density (we use the model from [**[mev2013]**](#References))

Using the cellwise density this becomes $N^V \in \mathbb{R}^{n_e \times n_x \times n_y}$
$$N^V_{eij} = \rho_{ij}c_{ije}\frac{\mathcal{N}_A}{A_e} $$
for convenience we expand this property and denote $\hat{N}^V \in \mathbb{R}^{n_k \times \hat{n}_x \times \hat{n}_y}$ where
$$ \hat{N}^V_{kij} = N^V_{epq}$$
where $e$ is the element, the k-ratio $k$ is characteristic to and $(p, q)$ the indices of the finite volume cells which lie inside the material subdomain $(i, j)$.

## Attenuation Coefficients
$$A \in \mathbb{R}^{\hat{n}_x \times \hat{n}_y \times n_k}$$
(see attenuation coeffs)

## Cross-Section
We denote the steps in $\epsilon$ with the index $t=1, ..., n_{\epsilon}$, such that $\epsilon_1 = \epsilon_{cutoff}$ and $\epsilon_{n_\epsilon} = \epsilon_{initial}$
$$\sigma^{emiss} \in \mathbb{R}^{n_k \times n_{\epsilon}}$$

**[TODO]** Maybe use this reference data: https://www-nds.iaea.org/epics/

## Electron Fluence 
$$ \psi^0  \in \mathbb{R}^{n_{\epsilon} \times \hat{n}_x \times \hat{n}_y}$$

## Trapezoidal Rule for Integration
Approximation of the integral by (fixed step size $\Delta x$, $x_0 = 0, x_1 = \Delta x, ...,  x_n = 1$)
$$ \int_0^1 f(x) dx = \sum_{i=0}^{n-1} \frac{f(x_i) + f(x_{i+1})}{2}\Delta x = (\frac{f(x_0)}{2} + \sum_{i=1}^{n-1} f(x_i)+\frac{f(x_n)}{2} ) \Delta x$$ 

## Implementation of the k-ratios


# References
**[mev2013]** N.Mevenkamp. Master thesis. Inverse Modeling in Electron Probe Microanalysis based on Deterministic Transport Equations (RWTH Aachen, 2013)
