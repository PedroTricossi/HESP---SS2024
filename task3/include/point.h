    #include "vec3d.h"

typedef struct point{
    /// Current position of the particle [m].
    t_vec3 cur_pos; 

    /// Current velocity of the particle [m/s].
    t_vec3 cur_vel;

    /// current accelaration of the particle [m/s^2].
    t_vec3 cur_acc;
} t_point;

// Initialize the particle
// TODO: Add inicialization by config file
void init_particle(t_point *particle);

// Update the particle position, velocity and acceleration using Verlet integration
void update(t_point *particle, double dt);
