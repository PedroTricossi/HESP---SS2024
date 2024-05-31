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
    float3* forces;
    Particle3D* particles;

    int numberOfThreads = 256;
    int numberOfBlocks = 32 * 46;

    cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    cudaMallocManaged(&forces, num_particles * sizeof(float3));

    t_neighbourList *nb_list = init_neighbourList(box_extension, cut_off_radious);

    
    
    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(i * 2, 10) ;
        float y = (i * 2 >= 10) ? fmod(floor(i * 4 / 10), 10): 0;
        float z = (i * 4 >= 100) ? fmod(floor(i * 8 / 100), 10) : 0;
        particles[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, nullptr, i);
        forces[i] = float3{ 0.0f, 0.0f, 0.0f };
    }

    

    for(int i = 0; i < num_particles; i++){
        add_particle(nb_list, &particles[i], cut_off_radious, box_extension);
    }

    std::cout << "num_particles: " << num_particles << std::endl;
    

    writeVTKFile(0, num_particles, particles);
    int step = 0;

    for (int step = 0; step < time_steps; ++step) {
        // Reset forces
        cudaMemset(forces, 0, num_particles * sizeof(float3));

        // std::cout << "particle: " << particles[1].getPosition().x << " " << particles[1].getPosition().y << " " << particles[1].getPosition().z << std::endl;

        // Compute forces using CUDA
        compute_force_between_particles <<< numberOfBlocks, numberOfThreads >>> (particles, forces, num_particles, eps, sigma, box_extension, cut_off_radious, nb_list);
        cudaDeviceSynchronize();

        // Integrate particles using CUDA
        apply_integrator_for_particle <<< numberOfBlocks, numberOfThreads >>> (particles, forces, num_particles, step_size, box_extension);
        cudaDeviceSynchronize();

        // Write the VTK file
        writeVTKFile(step + 1, num_particles, particles);

        // Clean the neighbour list
        clean_particle(nb_list);

        nb_list = init_neighbourList(box_extension, cut_off_radious);

        for(int i = 0; i < num_particles; i++){
            add_particle(nb_list, &particles[i], cut_off_radious, box_extension);
        }
    }

    cudaFree(particles);
    cudaFree(forces);
}

