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
    float num_cell_1d = box_extension / cut_off_radious;

    float x_pos = floor( particle->getPosition().x / cut_off_radious );
    float y_pos = floor( particle->getPosition().y / cut_off_radious );
    float z_pos = floor( particle->getPosition().z / cut_off_radious );

    int cell_index = fmod(x_pos + y_pos * num_cell_1d + z_pos * num_cell_1d * num_cell_1d, num_cell_1d * num_cell_1d * num_cell_1d);
	printf("add particle %d, to cell index: %d \n", particle->getId(), cell_index);
    for (int i = 0; i < cell_index; i++){
        current_cell = current_cell->next;
    }

    if(current_cell->particle == nullptr ){
        particle->setNextParticle(nullptr);

        current_cell->particle = particle;
        current_cell->num_particles++;

        return;
    }

    current_particle = current_cell->particle;

    while(current_particle->getNextParticle() != nullptr ){
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
        
        // printf("Cleaning cell: %d\n", current_cell->id);

        current_cell->particle = nullptr;
        cudaFree(current_cell);
        current_cell = next_cell;
    }
}


