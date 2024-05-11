#pragma once
#include "point.h"

typedef struct t_neighbourList{
    // id of the cell
    int id;
    
    // points to the next cell
    struct t_neighbourList *next;

    // points to the first particle in the cell
    t_point *particle;
    
    // number of particles in the cell
    int num_particles;
} t_neighbourList;

t_neighbourList *init_neighbourList(int size);

void add_particle(t_neighbourList *neighbourList, t_point *particle);

void clean_particle(t_neighbourList *neighbourList);

int detect_collision(t_neighbourList *neighbourList, t_point *particle);