#include <cstddef>
#include <math.h>
#include <iostream>
#include "../include/point.h"
#include "../include/vec3d.h"
#include "../include/constants.h"

void init_particle(t_point *particle, int *id){
    particle->id = *id;
    float x_pos=0., y_pos=0., z_pos=0.;

    x_pos = fmod(((CELL_SIZE / 2) + (CELL_SIZE * (*id))),  SIMULATION_SPACE * CELL_SIZE);

    if(*id >= SIMULATION_SPACE){
        int y_factor = ceil(*id / SIMULATION_SPACE);
        y_pos = fmod(((CELL_SIZE / 2) + (CELL_SIZE * y_factor)),  SIMULATION_SPACE * CELL_SIZE);
    }
    if(*id >= SIMULATION_SPACE * SIMULATION_SPACE){
        int z_factor = ceil(*id / (SIMULATION_SPACE * SIMULATION_SPACE));
        z_pos = fmod(((CELL_SIZE / 2) + (CELL_SIZE * z_factor)),  SIMULATION_SPACE * CELL_SIZE);
    }

    particle->cur_pos = {x_pos, y_pos, z_pos};

    particle->cur_vel = {50., 80., 0.};

    particle->cur_acc = {1., 0., 0.};

    particle->next = NULL;
}


t_point update(t_point *particle, double dt){
    t_point update_particle = *particle;
    t_vec3 new_pos, new_acc, new_vel;

    new_pos.x = particle->cur_pos.x + particle->cur_vel.x*dt + particle->cur_acc.x*(dt*dt*0.5);
    new_pos.y = particle->cur_pos.y + particle->cur_vel.y*dt + particle->cur_acc.y*(dt*dt*0.5);
    new_pos.z = particle->cur_pos.z + particle->cur_vel.z*dt + particle->cur_acc.z*(dt*dt*0.5);

    new_acc = particle->cur_acc;

    new_vel.x = particle->cur_vel.x + (particle->cur_acc.x + new_acc.x)*(dt * 0.5);
    new_vel.y = particle->cur_vel.y + (particle->cur_acc.y + new_acc.y)*(dt * 0.5);
    new_vel.z = particle->cur_vel.z + (particle->cur_acc.z + new_acc.z)*(dt * 0.5);

    update_particle.cur_pos = new_pos;
    update_particle.cur_vel = new_vel;
    update_particle.cur_acc = new_acc;

    return update_particle;
}

