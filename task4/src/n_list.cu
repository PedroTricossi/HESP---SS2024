#include "../include/n_list.cuh"
#include <iostream>
#include <cstddef>
#include <cmath>

t_neighbourList *init_neighbourList(float box_extension, float cut_off_radious){
    float num_cell_1d = box_extension / cut_off_radious;
    float num_cell_total = powf(num_cell_1d, 3);
    t_neighbourList *nb_list;

    cudaMallocManaged(&nb_list, sizeof(t_neighbourList));

    nb_list->next = nullptr ;
    nb_list->particle = nullptr ;
    nb_list->num_particles = 0;
    nb_list->id = 0;

    t_neighbourList *prev_cell = nb_list;

    for(int i = 1; i < num_cell_total; i++){
        t_neighbourList *new_cell;
        cudaMallocManaged(&new_cell, sizeof(t_neighbourList));

        new_cell->next = nullptr ;
        new_cell->particle = nullptr ;
        new_cell->num_particles = 0;
        new_cell->id = i;

        prev_cell->next = new_cell;
        prev_cell = new_cell;
    }

    return nb_list;
}

void add_particle(t_neighbourList *neighbourList, Particle3D *particle, float cut_off_radious, float box_extension){
    t_neighbourList *current_cell = neighbourList;
    Particle3D *current_particle;
    float x_pos = particle->getPosition().x;
    float y_pos = particle->getPosition().y;
    float z_pos = particle->getPosition().z;
    float num_cell_1d = box_extension / cut_off_radious;

    // std::cout << "x_pos: " << x_pos << " y_pos: " << y_pos << " z_pos: " << z_pos << std::endl;

    int cell_index = int(x_pos / cut_off_radious) + int(y_pos / cut_off_radious) * num_cell_1d + int(z_pos / cut_off_radious) * num_cell_1d * num_cell_1d;

    // std::cout << "cell_index: " << cell_index << " x_index: " << int(x_pos / cut_off_radious) << std::endl;

    for (int i = 0; i < cell_index; i++){
        current_cell = current_cell->next;
    }

    if(current_cell->particle == nullptr ){
        // std::cout << "Adding particle to cell: " << cell_index << std::endl;

        particle->setNextParticle(nullptr);

        current_cell->particle = particle;
        current_cell->num_particles++;

        return;
    }

    current_particle = current_cell->particle;

    while(current_particle->getNextParticle() != nullptr ){
        // std::cout << "current_particle:" << current_particle->getMass() << std::endl;

        current_particle = current_particle->getNextParticle();
    }
    
    // std::cout << "Adding particle to cell: " << cell_index << std::endl;
    particle->setNextParticle(nullptr);
    current_particle->setNextParticle(particle);
    current_cell->num_particles++;

    return;
}

void clean_particle(t_neighbourList *neighbourList){
    t_neighbourList *current_cell = neighbourList;

    while(current_cell != nullptr){
        // std::cout << "Cleaning cell: " << current_cell->id << std::endl;
        t_neighbourList *next_cell = current_cell->next;
        Particle3D *current_particle = current_cell->particle;

        while (current_particle != nullptr)
        {
            Particle3D *next_particle = current_particle->getNextParticle();
            current_particle->setNextParticle(nullptr);
            current_particle = next_particle;
 
        }
        
        current_cell->particle = nullptr;
        // free(current_cell);
        current_cell = next_cell;
    }
}


