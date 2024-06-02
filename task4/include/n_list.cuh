#pragma once
#include "particles.cuh"
class Particle3D;

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

__global__ void add_particles(t_neighbourList *neighbourList, Particle3D *particles, int num_particles, float cut_off_radious, float box_extension);

void add_particle(t_neighbourList *neighbourList, Particle3D *particle, float cut_off_radious, float box_extension);

void clean_particle(t_neighbourList *neighbourList);