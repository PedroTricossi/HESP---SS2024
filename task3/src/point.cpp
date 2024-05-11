#include "../include/point.h"
#include "../include/vec3d.h"

void init_particle(t_point *particle){
    particle->cur_pos = {0., 0., 0.};

    particle->cur_vel = {2., 0., 0.};

    particle->cur_acc = {1., 0., 0.};
}


void update(t_point *particle, double dt){
    t_vec3 new_pos, new_acc, new_vel;
    new_pos.x = particle->cur_pos.x + particle->cur_vel.x*dt + particle->cur_acc.x*(dt*dt*0.5);
    new_pos.y = particle->cur_pos.y + particle->cur_vel.y*dt + particle->cur_acc.y*(dt*dt*0.5);
    new_pos.z = particle->cur_pos.z + particle->cur_vel.z*dt + particle->cur_acc.z*(dt*dt*0.5);

    new_acc = particle->cur_acc;

    new_vel.x = particle->cur_vel.x + (particle->cur_acc.x + new_acc.x)*(dt * 0.5);
    new_vel.y = particle->cur_vel.y + (particle->cur_acc.y + new_acc.y)*(dt * 0.5);
    new_vel.z = particle->cur_vel.z + (particle->cur_acc.z + new_acc.z)*(dt * 0.5);

    particle->cur_pos = new_pos;
    particle->cur_vel = new_vel;
    particle->cur_acc = new_acc;
}

