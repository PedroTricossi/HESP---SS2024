#include "../include/n_list.cuh"
#include <iostream>
#include <cstddef>
#include <cmath>

t_neighbourList *init_neighbourList(float box_extension, float cut_off_radious){
    float num_cell_1d = box_extension / cut_off_radious;
    float num_cell_total = powf(num_cell_1d, 3);
    t_neighbourList *nb_list;

    // int deviceId;

    // cudaGetDevice(&deviceId);

    cudaMallocManaged(&nb_list, num_cell_total * sizeof(t_neighbourList));
    // cudaMemPrefetchAsync(nb_list,  num_cell_total * sizeof(t_neighbourList), deviceId);

    for(int i = 0; i < num_cell_total; i++){
        nb_list[i].next = nullptr ;
        nb_list[i].particle = nullptr ;
        nb_list[i].num_particles = 0;
        nb_list[i].id = i;

        if(i > 0){
            nb_list[i-1].next = &nb_list[i];
        }
    }

    return nb_list;
}

__global__ void add_particles(t_neighbourList *neighbourList, Particle3D *particles, int num_particles, float cut_off_radious, float box_extension){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_particles){
        // printf("Adding particle %d\n", index);
        __syncthreads();
        add_particle(neighbourList, &particles[index], cut_off_radious, box_extension);
        __syncthreads();
    }
}

__device__ void add_particle(t_neighbourList *neighbourList, Particle3D *particle, float cut_off_radious, float box_extension){
    t_neighbourList *current_cell;
    Particle3D *current_particle;
    float num_cell_1d = box_extension / cut_off_radious;
    int mutex = 0;

    float x_pos = floor( particle->getPosition().x / cut_off_radious );
    float y_pos = floor( particle->getPosition().y / cut_off_radious );
    float z_pos = floor( particle->getPosition().z / cut_off_radious );

    int cell_index = fmod(x_pos + y_pos * num_cell_1d + z_pos * num_cell_1d * num_cell_1d, num_cell_1d * num_cell_1d * num_cell_1d);

    // printf("Particle %d is in cell %d\n", particle->getId(), cell_index);

    particle->setNextParticle(nullptr);

    current_cell = &neighbourList[cell_index];

    while(mutex == 1){
        printf("Waiting\n");
    }
    atomicAdd(&mutex, -1);

    if(current_cell->particle == nullptr){

        printf("Particle %d is in cell %d\n", particle->getId(), cell_index);
        
        current_cell->particle = particle;

        atomicAdd(&current_cell->num_particles, 1);

        return;
    }


    current_particle = current_cell->particle;
    printf("Particle %d is in cell %d\n", particle->getId(), cell_index);
    while(current_particle->getNextParticle() != nullptr ){
        current_particle = current_particle->getNextParticle();
    }
    
    
    current_particle->setNextParticle(particle);
    atomicAdd(&current_cell->num_particles, 1);

    atomicAdd(&mutex, -1);

    return;

    return;
}

__global__ void clean_particle(t_neighbourList *neighbourList, float total_num_cells){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_num_cells){
        neighbourList[index].particle = nullptr;
        neighbourList[index].num_particles = 0;
    }
}


