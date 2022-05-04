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
import k_ratio_model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

elements = [physics.Iron(), physics.Manganese(), physics.Titan()]
x_rays = [physics.XRay(e, 1.) for e in elements]

## size of the vertical layer : 50 nm -> from -25nm to 25 nm
def build_material_and_parameters(beam_position_x, material_size_x):
    material = physics.Material(
        n_x = 450,
        n_y = 1,
        hat_n_x = 450,
        hat_n_y = 225,
        dim_x = [beam_position_x - material_size_x, beam_position_x + material_size_x],
        dim_y = [-material_size_x, 0.]
    )
    print(material.delta_x / nano)
    layer_interfaces = np.linspace(material.dim_x[0], material.dim_x[1], material.n_x + 1)
    assert(np.any(np.isclose(layer_interfaces / nano, 25)) or np.any(np.isclose(layer_interfaces / nano, -25)))
   
    # compute the layer centers:
    layer_centers = (layer_interfaces[1:] + layer_interfaces[:-1])/2.

    params = np.full((material.n_x, material.n_y, 2), 0.0)
    params[:, :, 0] = 1.0

    for i in range(material.n_x):
        if layer_centers[i] > -25.*nano and layer_centers[i] < 25.*nano:
            params[i, 0, :] = [0.0, 0.6]

    return material, params

mat, par = build_material_and_parameters(0., 25/5 * 450/2*nano)


k_ratios = []
for idx, e_beam_position in enumerate(np.linspace(-25./5.*100.*nano, 25./5.*100.*nano, 101)):
    material, true_parameters = build_material_and_parameters(e_beam_position, 25./5. * 450/2*nano)

    ## this detector position fakes a 40° take-off angle (the nm shift in the material does not influence the detector position)
    detector = physics.Detector(
        x=1.,
        y=1.*np.tan(0.6981),
        material = material)

    eps_initial_keV = 15.5
    eps_cutoff_keV = 4.5
    n_epsilon = 550

    beam_energy_keV = 15.

    ## BEAM VARIANCE -> 50.nano std_dev means, that ~94 percent of the beams energy is inside of 4*50nano (meters)
    beam_size_x = (50.*nano)**2

    electron_beam = physics.ElectronBeam(
            size=[beam_size_x, None],
            pos=[e_beam_position, None],
            beam_energy_keV=beam_energy_keV,
            energy_variation_keV=(0.2)**2
        )

    e = experiment.Experiment(
            material=material,
            detector=detector,
            electron_beam=electron_beam,
            elements=elements,
            x_ray_transitions=x_rays,
            epsilon_initial_keV=eps_initial_keV,
            epsilon_cutoff_keV=eps_cutoff_keV,
            n_epsilon=n_epsilon
    )

    mass_fractions = experiment.mass_fractions_from_parameters(material.n_x, material.n_y, true_parameters)
    #test = m1model.solve_forward(e, mass_fractions)

    #break
    k_ratios_ = experiment.k_ratios(e, true_parameters)
    k_ratios.append(k_ratios_)

plt.imshow(test["solution"][100, 0, :, :]); plt.show()

k_ratios_np = [np.array(k_r) for k_r in k_ratios]

with open('experiments/FeMnTi_15_new_density.pkl', 'wb') as writefile:
    pickle.dump(k_ratios_np, writefile)

k_r_0 = [k_r[0] for k_r in k_ratios_np]
k_r_1 = [k_r[1] for k_r in k_ratios_np]
k_r_2 = [k_r[2] for k_r in k_ratios_np]
plt.plot(k_r_0)
plt.plot(k_r_1)
plt.plot(k_r_2)
plt.show()

k_r_0

## compute standard MnTi

k_ratios_std = []

for std in ["MnTi", "Fe"]:
    if std == "Fe":
        material = physics.Material(
            n_x = 450,
            n_y = 1,
            hat_n_x = 450,
            hat_n_y = 225,
            dim_x = [- 25/5 * 450/2*nano, 25/5 * 450/2*nano],
            dim_y = [-25/5 * 450/2*nano, 0.]
        )
    elif std == "MnTi":
        material = physics.Material(
            n_x = 450,
            n_y = 1,
            hat_n_x = 450,
            hat_n_y = 225,
            dim_x = [- 25/5 * 450/2*nano, 25/5 * 450/2*nano],
            dim_y = [-25/5 * 450/2*nano, 0.]
        )
    true_parameters = np.full((material.n_x, material.n_y, 2), 0.0)
    if std == "Fe":
        true_parameters[:, :, 0] = 1.0 ## parameter for Fe
    elif std == "MnTi":
        true_parameters[:, :, 1] = 0.8

    detector = physics.Detector(
        x=1.,
        y=1.*np.tan(0.6981),
        material = material)
    
    eps_initial_keV = 15.5
    eps_cutoff_keV = 4.5
    n_epsilon = 550

    beam_energy_keV = 15.0

    beam_size_x = (50.*nano)**2

    electron_beam = physics.ElectronBeam(
            size=[beam_size_x, None],
            pos=[0., None],
            beam_energy_keV=beam_energy_keV,
            energy_variation_keV=(0.2)**2
        )

    e = experiment.Experiment(
            material=material,
            detector=detector,
            electron_beam=electron_beam,
            elements=elements,
            x_ray_transitions=x_rays,
            epsilon_initial_keV=eps_initial_keV,
            epsilon_cutoff_keV=eps_cutoff_keV,
            n_epsilon=n_epsilon
    )

    mass_fractions = experiment.mass_fractions_from_parameters(material.n_x, material.n_y, true_parameters)
    #test = m1model.solve_forward(e, mass_fractions)

    #break
    k_ratios_ = experiment.k_ratios(e, true_parameters)
    k_ratios_std.append(k_ratios_)

with open('experiments/FeMnTi.pkl', 'rb') as readfile:
    k_ratios_np = pickle.load(readfile)

k_r_0 = [k_r[0] / float(k_ratios_std[1][0]) for k_r in k_ratios_np]
k_r_1 = [k_r[1] / float(k_ratios_std[0][1]) for k_r in k_ratios_np]
k_r_2 = [k_r[2] / float(k_ratios_std[0][2]) for k_r in k_ratios_np]
plt.plot(k_r_0)
plt.plot(k_r_1)
plt.plot(k_r_2)
plt.show()