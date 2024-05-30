#pragma once
#include "particles.cuh"

typedef struct t_neighbourList{
    // id of the cell
    int id;
    
    // points to the next cell
    struct t_neighbourList *next;

    // points to the first particle in the cell
    Particle3D *particle;
    
    // number of particles in the cell
    int num_particles;
} t_neighbourList;


t_neighbourList *init_neighbourList(float box_extension, float cut_off_radious);

void add_particle(t_neighbourList *neighbourList, Particle3D *particle, float cut_off_radious, float box_extension);

void clean_particle(t_neighbourList *neighbourList);

int detect_collision(t_neighbourList *neighbourList, Particle3D *particle);