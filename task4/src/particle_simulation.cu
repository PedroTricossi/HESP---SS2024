#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/particles.cuh"

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


void start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float box_extension){
    int deviceId;
    float3* forces;
    Particle3D* particles;
        
    cudaGetDevice(&deviceId);

    int numberOfThreads = 256;
    int numberOfBlocks = 32;

    cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    cudaMallocManaged(&forces, num_particles * sizeof(float3));
    
    std::cout << "BOX: " << box_extension << std::endl;
    
    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(i, 10);
        float y = (i >= 10) ? fmod(floor(i / 10), 10): 0;
        float z = (i >= 100) ? fmod(floor(i / 100), 10) : 0;
        particles[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f);
        forces[i] = float3{ 0.0f, 0.0f, 0.0f };
    }

    writeVTKFile(0, num_particles, particles);

    for (int step = 0; step < time_steps; ++step) {
        // Reset forces
        cudaMemset(forces, 0, num_particles * sizeof(float3));

        // Compute forces using CUDA
        compute_force_between_particles <<< numberOfBlocks, numberOfThreads >>> (particles, forces, num_particles, eps, sigma, box_extension);
        cudaDeviceSynchronize();

        // Integrate particles using CUDA
        apply_integrator_for_particle <<< numberOfBlocks, numberOfThreads >>> (particles, forces, num_particles, step_size, box_extension);
        cudaDeviceSynchronize();

        // Write the VTK file
        writeVTKFile(step + 1, num_particles, particles);
    }

    cudaFree(particles);
    cudaFree(forces);
}

