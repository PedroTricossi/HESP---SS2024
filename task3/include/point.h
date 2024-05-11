#pragma once
#include "vec3d.h"

typedef struct t_point{
    /// Unique identifier of the particle.
    int id;

    /// Current position of the particle [m].
    t_vec3 cur_pos; 

    /// Current velocity of the particle [m/s].
    t_vec3 cur_vel;

    /// current accelaration of the particle [m/s^2].
    t_vec3 cur_acc;

    /// next particle in the same cell
    t_point *next;
} t_point;

// Initialize the particle
// TODO: Add inicialization by config file
void init_particle(t_point *particle, int *id);

// Update the particle position, velocity and acceleration using Verlet integration
t_point update(t_point *particle, double dt);

