# Mass Fractions

## Material Subdomains
The material is subdivided in $(n_x, n_y)$ subdomains (cells). The mass fracions are defined as a piecewise constant function over the subdomain.

## Mass Fractions of the Subdomains
Mass fractions for each element $(n_e)$ in each material cell(subdomain) $(n_x \times n_y)$
$$ c \in \mathbb{R}^{n_x \times n_y \times n_e}$$

# Density
$$ \rho = \sum_{e=1}^{n_e} c_e \bar{\rho}_e $$
where 
$\bar{\rho}_e$ is the elemental density of element $e$.

## Densities of the Subdomains
Density in each material cell(subdomain) $(n_x \times n_y)$
$$ \rho \in \mathbb{R}^{n_x \times n_y}$$
$$ \rho_{ij} = \sum_{e=1}^{n_e} c_{ije}\bar{\rho}_e$$

## Derivative of the Density
$$ \frac{\partial \rho_{ij}}{\partial c_{mne}} = \begin{cases} \bar{\rho}_e & i=m \land j=n \\ 0 & else \end{cases}$$


 **[TODO]** Future implementation:
$$ \frac{1}{\rho_{ij}} = \sum_{e=1}^{n_e} \frac{c_{ije}}{\bar{\rho}_e}$$

# Weighted Material Property
Many material properties (used in this implementation) can be written as (generically for $S \in \mathbb{R}^{n_p \times n_x \times n_y}$)

$$ S_{pij} = \sum_{e=1}^{n_e} c_{ije} \rho_{ij} \bar{S}_{pe}$$
where
$\bar{S}_{pe}$ is the elemental (specific) material property.

## Derivative of the Property
$$ \frac{\partial S_{ij}}{\partial c_{mnq}} = \begin{cases} \rho_{ij} \bar{S}_q + \bar{\rho}_q \sum_{e=1}^{n_e} \bar{S}_e c_{ije} & i=m \land j=n \\
0 & else\end{cases} $$

**[TODO]** use the other density description and add multi-dimensional material properties
