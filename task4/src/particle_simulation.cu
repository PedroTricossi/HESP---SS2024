#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/particles.cuh"
#include "../include/n_list.cuh"

void writeVTKFile(int step, int num_particles, Particle3D* particles) {
    std::ofstream simulationFile("simulation_" + std::to_string(step) + ".vtk");

    simulationFile << "# vtk DataFile Version 3.0 \n";
    simulationFile << "Lennard-Jones particle simulation \n";
    simulationFile << "ASCII \n";
    simulationFile << "DATASET UNSTRUCTURED_GRID \n";
    simulationFile << "POINTS " << num_particles << " float \n";

    for (int i = 0; i < num_particles; i++) {
        float3 pos = particles[i].getPosition();
        simulationFile << pos.x << " " << pos.y << " " << pos.z << "\n";
    }

    simulationFile << "CELLS " << "0" << " " << "0" << "\n";
    simulationFile << "CELL_TYPES " << "0" << "\n";
    simulationFile << "POINT_DATA " << num_particles << "\n";
    simulationFile << "SCALARS mass float \n";
    simulationFile << "LOOKUP_TABLE default \n";

    for (int i = 0; i < num_particles; i++) {
        simulationFile << particles[i].getMass() << "\n";
    }

    simulationFile << "VECTORS velocity float \n";
    for (int i = 0; i < num_particles; i++) {
        float3 vel = particles[i].getVelocity();
        simulationFile << vel.x << " " << vel.y << " " << vel.z << "\n";
    }
}


void start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float box_extension, float cut_off_radious){
    float3 *forces;
    Particle3D* particles;

    int numberOfThreads = 256;
    int numberOfBlocks = 32 * 46;
    int deviceId;

    float num_cell_1d = box_extension / cut_off_radious;
    float num_cell_total = powf(num_cell_1d, 3);

    cudaGetDevice(&deviceId);

    cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    cudaMemPrefetchAsync(particles,  num_particles * sizeof(Particle3D), deviceId);

    cudaMallocManaged(&forces, num_particles * sizeof(float3));
    cudaMemPrefetchAsync(forces, num_particles * sizeof(float3), deviceId);

    // t_neighbourList *nb_list = init_neighbourList(box_extension, cut_off_radious);

    t_neighbourList *nb_list = nullptr;

    int pos = 0;    
    
    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(pos, box_extension) ;
        float y = (pos >= box_extension) ? fmod(floor(pos * 2 / box_extension), box_extension): 0;
        float z = (pos * 4 >= box_extension * box_extension) ? fmod(floor((pos * 4) / (box_extension * box_extension) ), box_extension) : 0;
        particles[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, nullptr, i);
        forces[i] = float3{ 0.0f, 0.0f, 0.0f };

        pos += (cut_off_radious);
    }

    // add_particles<<< numberOfBlocks, numberOfThreads >>>(nb_list, particles, num_particles, cut_off_radious, box_extension);
    // cudaDeviceSynchronize();

    // for(int i = 0; i < num_particles; i++){
    //     printf("cell %d has %d particles\n",i, nb_list[i].num_particles);
    // }
    

    std::cout << num_particles << ", ";
    

    // writeVTKFile(0, num_particles, particles);
    int step = 0;

    for (int step = 0; step < time_steps; ++step) {
        cudaMemset(forces, 0, num_particles * sizeof(float3));

        // std::cout << "particle: " << particles[1].getPosition().x << " " << particles[1].getPosition().y << " " << particles[1].getPosition().z << std::endl;
        // printf("step: %d\n", step);
        

        // Compute forces using CUDA
        compute_force_between_particles <<< numberOfBlocks, numberOfThreads >>> (particles, forces, num_particles, eps, sigma, box_extension, cut_off_radious, nb_list);
        cudaDeviceSynchronize();

        // // Integrate particles using CUDA
        apply_integrator_for_particle <<< numberOfBlocks, numberOfThreads >>> (particles, forces, num_particles, step_size, box_extension);
        cudaDeviceSynchronize();

        // // Write the VTK file
        // writeVTKFile(step + 1, num_particles, particles);

        // // Clean the neighbour list
        // clean_particle<<< numberOfBlocks, numberOfThreads >>>(nb_list, num_cell_total);
        // cudaDeviceSynchronize();
        
        // // nb_list = init_neighbourList(box_extension, cut_off_radious);

        // add_particles<<< numberOfBlocks, numberOfThreads >>>(nb_list, particles, num_particles, cut_off_radious, box_extension);
        // cudaDeviceSynchronize();
    }

    cudaFree(particles);
    cudaFree(forces);
}
