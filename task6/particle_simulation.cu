#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include "particles.cuh"
// #include "../include/n_list.cuh"

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

    simulationFile << "SCALARS radius float \n";
    simulationFile << "LOOKUP_TABLE default \n";

    for (int i = 0; i < num_particles; i++) {
        simulationFile << particles[i].getRadius() << "\n";
    }

    simulationFile << "VECTORS velocity float \n";
    for (int i = 0; i < num_particles; i++) {
        float3 vel = particles[i].getVelocity();
        simulationFile << vel.x << " " << vel.y << " " << vel.z << "\n";
    }
}


void start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity,float box_extension, float cut_off_radious)
{
    float3 *ha_forces;
    float3 *hb_forces;
    float3 *da_forces;
    float3 *db_forces;
    Particle3D* particles;
    int deviceId;
    cudaDeviceProp prop;

    float num_cell_1d = box_extension / cut_off_radious;
    float num_cell_total = powf(num_cell_1d, 3);

    cudaGetDevice(&deviceId);

    cudaGetDeviceProperties(&prop, deviceId);

    int numberOfThreads = 256;
    int numberOfBlocks = 32 * prop.multiProcessorCount;


    // cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    // cudaMemPrefetchAsync(particles,  num_particles * sizeof(Particle3D), deviceId);

    // cudaMallocManaged(&forces, num_particles * sizeof(float3));
    // cudaMemPrefetchAsync(forces, num_particles * sizeof(float3), deviceId);

    Particle3D *h_a = NULL;
	Particle3D *h_b = NULL;

	cudaMallocHost((void**)&h_a, sizeof(Particle3D) * (num_particles / 2));
	cudaMallocHost((void**)&h_b, sizeof(Particle3D) * (num_particles / 2));
    cudaMallocHost((void**)&ha_forces, sizeof(float3) * (num_particles / 2));
    cudaMallocHost((void**)&hb_forces, sizeof(float3) * (num_particles / 2));

    Particle3D* d_a = NULL; 
	Particle3D* d_b = NULL; 

	cudaMalloc((void**)&d_a, sizeof(Particle3D) * (num_particles / 2));
	cudaMalloc((void**)&d_b, sizeof(Particle3D) * (num_particles / 2));
    cudaMalloc((void**)&da_forces, sizeof(float3) * (num_particles / 2));
    cudaMalloc((void**)&db_forces, sizeof(float3) * (num_particles / 2));

    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&stream[i]);
    }

    t_neighbourList *nb_list = nullptr;

    int pos = 0;    
    
    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(pos, box_extension) ;
        float y = (pos >= box_extension) ? fmod(floor(pos * 2 / box_extension), box_extension): 0;
        float z = (pos * 4 >= box_extension * box_extension) ? fmod(floor((pos * 4) / (box_extension * box_extension) ), box_extension) : 0;
        
        if(i < (num_particles / 2))
            h_a[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
        else
            h_b[i - (num_particles / 2)] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
        
        if (i < num_particles / 2)
            ha_forces[i] = float3{ 0.0f, 0.0f, 0.0f };
        else
            hb_forces[i - (num_particles / 2)] = float3{ 0.0f, 0.0f, 0.0f };

        pos += 2;
    }

    std::cout << num_particles << ", ";

    cudaMemcpyAsync(d_a, h_a, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(d_b, h_b, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);

    cudaMemcpyAsync(da_forces, ha_forces, sizeof(float3)* (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(db_forces, hb_forces, sizeof(float3)* (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);

    

    // writeVTKFile(0, num_particles, particles);

    for (int step = 0; step < time_steps; ++step) {
        cudaMemset(da_forces, 0, sizeof(float3)* (num_particles / 2));
        cudaMemset(db_forces, 0, sizeof(float3)* (num_particles / 2));

        // Compute forces using CUDA

        compute_force_between_particles <<< numberOfBlocks, numberOfThreads, 0, stream[0]>>> (d_a, da_forces, num_particles, eps, sigma, k_n, gamma, gravity, box_extension, cut_off_radious, nb_list);
        compute_force_between_particles <<< numberOfBlocks, numberOfThreads, 0, stream[1]>>> (d_b, db_forces, num_particles, eps, sigma, k_n, gamma, gravity, box_extension, cut_off_radious, nb_list);
                
        // Integrate particles using CUDA
        apply_integrator_for_particle_euler <<< numberOfBlocks, numberOfThreads, 0, stream[0]>>> (d_a, da_forces, num_particles, step_size, box_extension);
        apply_integrator_for_particle_euler <<< numberOfBlocks, numberOfThreads, 0, stream[1]>>> (d_b, db_forces, num_particles, step_size, box_extension);
        cudaDeviceSynchronize();

        // Write the VTK file
        // writeVTKFile(step + 1, num_particles, particles);
    }

    // cudaFree(particles);
    // cudaFree(forces);
}

/*particle simulation using graphs
void start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity, float box_extension, float cut_off_radious) {
    float3 *ha_forces;
    float3 *hb_forces;
    float3 *da_forces;
    float3 *db_forces;
    Particle3D* particles;
    int deviceId;
    cudaDeviceProp prop;

    float num_cell_1d = box_extension / cut_off_radious;
    float num_cell_total = powf(num_cell_1d, 3);

    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);

    int numberOfThreads = 256;
    int numberOfBlocks = (num_particles / 2 + numberOfThreads - 1) / numberOfThreads;

    Particle3D *h_a = NULL;
    Particle3D *h_b = NULL;

    cudaMallocHost((void**)&h_a, sizeof(Particle3D) * (num_particles / 2));
    cudaMallocHost((void**)&h_b, sizeof(Particle3D) * (num_particles / 2));
    cudaMallocHost((void**)&ha_forces, sizeof(float3) * (num_particles / 2));
    cudaMallocHost((void**)&hb_forces, sizeof(float3) * (num_particles / 2));

    Particle3D* d_a = NULL;
    Particle3D* d_b = NULL;

    cudaMalloc((void**)&d_a, sizeof(Particle3D) * (num_particles / 2));
    cudaMalloc((void**)&d_b, sizeof(Particle3D) * (num_particles / 2));
    cudaMalloc((void**)&da_forces, sizeof(float3) * (num_particles / 2));
    cudaMalloc((void**)&db_forces, sizeof(float3) * (num_particles / 2));

    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&stream[i]);
    }

    t_neighbourList *nb_list = nullptr;

    int pos = 0;    

    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(pos, box_extension);
        float y = (pos >= box_extension) ? fmod(floor(pos * 2 / box_extension), box_extension) : 0;
        float z = (pos * 4 >= box_extension * box_extension) ? fmod(floor((pos * 4) / (box_extension * box_extension)), box_extension) : 0;
        
        if(i < (num_particles / 2))
            h_a[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
        else
            h_b[i - (num_particles / 2)] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
        
        if (i < num_particles / 2)
            ha_forces[i] = float3{ 0.0f, 0.0f, 0.0f };
        else
            hb_forces[i - (num_particles / 2)] = float3{ 0.0f, 0.0f, 0.0f };

        pos += 2;
    }

    cudaMemcpyAsync(d_a, h_a, sizeof(Particle3D) * (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_b, h_b, sizeof(Particle3D) * (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(da_forces, ha_forces, sizeof(float3) * (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(db_forces, hb_forces, sizeof(float3) * (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);

    cudaGraph_t graph[2];
    cudaGraphExec_t instance[2];

    // Create and capture graphs for both streams
    for (int i = 0; i < 2; i++) {
        cudaStreamBeginCapture(stream[i], cudaStreamCaptureModeGlobal);

        // Reset forces
        cudaMemset(i == 0 ? da_forces : db_forces, 0, sizeof(float3) * (num_particles / 2));

        // Compute forces using CUDA
        compute_force_between_particles<<<numberOfBlocks, numberOfThreads, 0, stream[i]>>>(
            i == 0 ? d_a : d_b,
            i == 0 ? da_forces : db_forces,
            num_particles / 2,
            eps,
            sigma,
            k_n,
            gamma,
            gravity,
            box_extension,
            cut_off_radious,
            nb_list
        );

        // Integrate particles using CUDA
        apply_integrator_for_particle_euler<<<numberOfBlocks, numberOfThreads, 0, stream[i]>>>(
            i == 0 ? d_a : d_b,
            i == 0 ? da_forces : db_forces,
            num_particles / 2,
            step_size,
            box_extension
        );

        cudaStreamEndCapture(stream[i], &graph[i]);
        cudaGraphInstantiate(&instance[i], graph[i], NULL, NULL, 0);
    }

    for (int step = 0; step < time_steps; ++step) {
        for (int i = 0; i < 2; i++) {
            cudaGraphLaunch(instance[i], stream[i]);
        }

        // Synchronize streams
        for (int i = 0; i < 2; i++) {
            cudaStreamSynchronize(stream[i]);
        }

        // Optional: Write the VTK file
        // writeVTKFile(step + 1, num_particles, particles);
    }

    // Cleanup
    for (int i = 0; i < 2; i++) {
        cudaGraphDestroy(graph[i]);
        cudaGraphExecDestroy(instance[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(da_forces);
    cudaFree(db_forces);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(ha_forces);
    cudaFreeHost(hb_forces);
}


*/
