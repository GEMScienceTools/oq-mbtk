# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8


import numpy as np
from numba import jit

from .fastmath import spmat_multivec_mul
from .simulated_annealing import add_weights_to_matrices


def evaluate_fitness(G, d, x):
    residuals = G @ x - d
    return np.sum(residuals**2)


def evaluate_swarm_fitness(G, d, swarm_pos):
    swarm_predictions = spmat_multivec_mul(G, swarm_pos)
    swarm_residuals = swarm_predictions - d

    swarm_fitness = np.sum(swarm_residuals**2, axis=1)
    return swarm_fitness


# @jit
def update_particle_velocity(
    swarm_pos,
    vels,
    best_swarm_pos,
    global_best_pos,
    intertia,
    cognitive_coeff,
    social_coeff,
    min=None,
    max=None,
):
    # Update particle velocity
    r1 = np.random.uniform(-0.25, 1, swarm_pos.shape[0])
    r2 = np.random.uniform(-0.25, 1, swarm_pos.shape[0])

    particles_at_bounds = np.logical_or(swarm_pos == min, swarm_pos == max)
    vels[particles_at_bounds] = -1.0 * vels[particles_at_bounds]

    vels = (
        intertia * vels
        + cognitive_coeff * r1 * (best_swarm_pos - swarm_pos)
        + social_coeff * r2 * (global_best_pos - swarm_pos)
    )

    return vels


@jit
def update_particle_position(x_i, v_i, min_bounds, max_bounds):
    x_i += v_i

    # Clip position to stay within bounds
    x_i = np.clip(x_i, min_bounds, max_bounds)

    return x_i


def lls_particle_swarm(
    A,
    d,
    bounds,
    x0=None,
    weights=None,
    swarm_size=50,
    max_iterations=100,
    inertia=0.7,
    cognitive_coeff=0.1,
    social_coeff=0.1,
    tol=1e-6,
    print_updates="update",
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    num_variables = x0.shape[0]

    # Extract minimum and maximum bounds for each variable
    min_bounds = bounds[0]
    max_bounds = bounds[1]

    Asp, dw = add_weights_to_matrices(A, d, weights)

    # Initialize swarm
    swarm_pos = np.zeros((swarm_size, num_variables))
    swarm_vel = np.zeros((swarm_size, num_variables))
    swarm_best_pos = np.zeros((swarm_size, num_variables))
    swarm_best_fitness = np.full(swarm_size, np.inf)
    global_best_pos = np.zeros(num_variables)
    global_best_fitness = np.inf

    global_best_fitnesses = np.zeros(max_iterations)

    # Initialize particles
    for i in range(swarm_size):
        swarm_pos[i] = x0  # + np.exp(
        #    np.random.uniform(
        #        np.log(min_bounds), np.log(max_bounds), num_variables
        #    )
        # )

        swarm_pos[i] = np.clip(swarm_pos[i], min_bounds, max_bounds)

        swarm_vel[i] = np.random.randn(num_variables) * 1e-3

    # Main optimization loop
    for iteration in range(max_iterations):
        swarm_fitness = evaluate_swarm_fitness(Asp, dw, swarm_pos)

        better_fits = swarm_fitness < swarm_best_fitness
        swarm_best_fitness[better_fits] = swarm_fitness[better_fits]
        swarm_best_pos[better_fits] = swarm_pos[better_fits]

        global_best = np.argmin(swarm_best_fitness)

        str_end = "\r" if iteration < max_iterations - 1 else "\n"
        status_string = (
            "current norm: "
            + f"{format_engineering_notation(global_best_fitness)},"
            + f"{iteration}/{max_iterations}"
        )

        if swarm_best_fitness[global_best] < global_best_fitness:
            global_best_fitness = swarm_best_fitness[global_best]
            global_best_pos = swarm_best_pos[global_best]

            if print_updates == "update":
                print(status_string, end=str_end)

        # for i in range(swarm_size):
        #    # Evaluate fitness
        #    # fitness = evaluate_fitness(G, d, swarm_pos[i])
        #    fitness = swarm_fitness[i]

        #    # Update particle's best position and fitness
        #    # if fitness < swarm_best_fitness[i]:
        #    #    swarm_best_pos[i] = swarm_pos[i]
        #    #    swarm_best_fitness[i] = fitness

        #    # Update global best position and fitness
        #    if fitness < global_best_fitness:
        #        global_best_pos = swarm_pos[i]
        #        global_best_fitness = fitness

        if print_updates == "iter":
            print(status_string, end=str_end)
        global_best_fitnesses[iteration] = global_best_fitness

        # Termination condition based on tolerance
        if global_best_fitness < tol:
            break

        # Update particle velocities and positions
        for i in range(swarm_size):
            swarm_vel[i] = update_particle_velocity(
                swarm_pos[i],
                swarm_vel[i],
                swarm_best_pos[i],
                global_best_pos,
                inertia,
                cognitive_coeff,
                social_coeff,
            )
            swarm_pos[i] = update_particle_position(
                swarm_pos[i], swarm_vel[i], min_bounds, max_bounds
            )

    print("norm", global_best_fitness)
    return global_best_pos, global_best_fitnesses


def format_engineering_notation(number, num_characters=10):
    try:
        # Get the exponent of the number in engineering notation
        exponent = int(np.floor(np.log10(abs(number))) // 3) * 3

        # Format the number in engineering notation with fixed width
        formatted_number = f"{number / (10 ** exponent):.{num_characters-1}f}"

        # Construct the final string with the exponent
        final_string = f"{formatted_number}e{exponent}"
    except:
        final_string = "NaN"

    return final_string
