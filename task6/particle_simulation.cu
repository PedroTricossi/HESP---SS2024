#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include "particles.cuh"
// #include "../include/n_list.cuh"

void writeVTKFile(int step, int num_particles, Particle3D* particles_steam_1, Particle3D* particles_steam_2) {
    std::ofstream simulationFile("simulation_" + std::to_string(step) + ".vtk");

    simulationFile << "# vtk DataFile Version 3.0 \n";
    simulationFile << "Lennard-Jones particle simulation \n";
    simulationFile << "ASCII \n";
    simulationFile << "DATASET UNSTRUCTURED_GRID \n";
    simulationFile << "POINTS " << num_particles  << " float \n";

    for (int i = 0; i < (num_particles / 2); i++) {
        float3 pos = particles_steam_1[i].getPosition();
        simulationFile << pos.x << " " << pos.y << " " << pos.z << "\n";
    }

    for (int i = 0; i < (num_particles / 2); i++) {
        float3 pos = particles_steam_2[i].getPosition();
        simulationFile << pos.x << " " << pos.y << " " << pos.z << "\n";
    }

    simulationFile << "CELLS " << "0" << " " << "0" << "\n";
    simulationFile << "CELL_TYPES " << "0" << "\n";
    simulationFile << "POINT_DATA " << (num_particles) << "\n";
    simulationFile << "SCALARS mass float \n";
    simulationFile << "LOOKUP_TABLE default \n";

    for (int i = 0; i < (num_particles / 2); i++) {
        simulationFile << particles_steam_1[i].getMass() << "\n";
    }

    for (int i = 0; i < (num_particles / 2); i++) {
        simulationFile << particles_steam_2[i].getMass() << "\n";
    }

    simulationFile << "SCALARS radius float \n";
    simulationFile << "LOOKUP_TABLE default \n";

    for (int i = 0; i < (num_particles / 2); i++) {
        simulationFile << particles_steam_1[i].getRadius() << "\n";
    }

    for (int i = 0; i < (num_particles / 2); i++) {
        simulationFile << particles_steam_2[i].getRadius() << "\n";
    }

    simulationFile << "VECTORS velocity float \n";
    for (int i = 0; i < (num_particles / 2); i++) {
        float3 vel = particles_steam_1[i].getVelocity();
        simulationFile << vel.x << " " << vel.y << " " << vel.z << "\n";
    }

    for (int i = 0; i < (num_particles / 2); i++) {
        float3 vel = particles_steam_2[i].getVelocity();
        simulationFile << vel.x << " " << vel.y << " " << vel.z << "\n";
    }
}


void start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity,float box_extension, float cut_off_radious)
{
    // define forces and particles
    float3 *host_forces_a;
    float3 *host_forces_b;
    float3 *device_forces_a;
    float3 *device_forces_b;
    
    Particle3D *host_particles_a = NULL;
	Particle3D *host_particles_b = NULL;
    Particle3D* device_particles_a = NULL; 
	Particle3D* device_particles_b = NULL; 

    int deviceId;
    cudaDeviceProp prop;

    float num_cell_1d = box_extension / cut_off_radious;
    float num_cell_total = powf(num_cell_1d, 3);

    cudaGetDevice(&deviceId);

    cudaGetDeviceProperties(&prop, deviceId);

    int numberOfThreads = 256;
    int numberOfBlocks = 32 * prop.multiProcessorCount;


    // allocate memory for particles and forces
	cudaMallocHost((void**)&host_particles_a, sizeof(Particle3D) * (num_particles / 2));
	cudaMallocHost((void**)&host_particles_b, sizeof(Particle3D) * (num_particles / 2));
    cudaMallocHost((void**)&host_forces_a, sizeof(float3) * (num_particles / 2));
    cudaMallocHost((void**)&host_forces_b, sizeof(float3) * (num_particles / 2));


	cudaMalloc((void**)&device_particles_a, sizeof(Particle3D) * (num_particles / 2));
	cudaMalloc((void**)&device_particles_b, sizeof(Particle3D) * (num_particles / 2));
    cudaMalloc((void**)&device_forces_a, sizeof(float3) * (num_particles / 2));
    cudaMalloc((void**)&device_forces_b, sizeof(float3) * (num_particles / 2));

    // create streams
    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // initiate particles
    int pos = 4;    
    
    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(pos, box_extension) ;
        float y = (pos >= box_extension) ? fmod(floor(pos * 2 / box_extension), box_extension): 0;
        float z = (pos * 4 >= box_extension * box_extension) ? fmod(floor((pos * 4) / (box_extension * box_extension) ), box_extension) : 0;
        
        if(i < (num_particles / 2))
            host_particles_a[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
            
            // if (i == 42) {
            //     host_particles_a[i] = Particle3D(float3{ x, y, z }, float3{ -2.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
            // }
        else
            host_particles_b[i - (num_particles / 2)] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
        
        if (i < num_particles / 2)
            host_forces_a[i] = float3{ 0.0f, 0.0f, 0.0f };
        else
            host_forces_b[i - (num_particles / 2)] = float3{ 0.0f, 0.0f, 0.0f };

        pos += 2;
    }

    std::cout << num_particles << ", ";

    // Write the particles initial state
    writeVTKFile(0, num_particles, host_particles_a, host_particles_b);

    // copy particles and forces to device
    cudaMemcpyAsync(device_particles_a, host_particles_a, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(device_particles_b, host_particles_b, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);

    cudaMemcpyAsync(device_forces_a, host_forces_a, sizeof(float3)* (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(device_forces_b, host_forces_b, sizeof(float3)* (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);


    for (int step = 0; step < time_steps; ++step) {
        cudaMemset(device_forces_a, 0, sizeof(float3)* (num_particles / 2));
        cudaMemset(device_forces_b, 0, sizeof(float3)* (num_particles / 2));

        // Compute forces using CUDA streams
        compute_force_between_particles <<< numberOfBlocks, numberOfThreads, 0, stream[0]>>> (device_particles_a, device_forces_a, num_particles, eps, sigma, k_n, gamma, gravity, box_extension, cut_off_radious);
        compute_force_between_particles <<< numberOfBlocks, numberOfThreads, 0, stream[1]>>> (device_particles_b, device_forces_b, num_particles, eps, sigma, k_n, gamma, gravity, box_extension, cut_off_radious);
        cudaDeviceSynchronize();

        // compute colision between particles in diferent streams
        compute_colision_between_streams <<< numberOfBlocks, numberOfThreads>>> (device_particles_a, device_particles_b, device_forces_a, device_forces_b, num_particles, k_n, gamma, box_extension, cut_off_radious);
        cudaDeviceSynchronize();

                
        // Integrate particles using CUDA
        apply_integrator_for_particle_euler <<< numberOfBlocks, numberOfThreads, 0, stream[0]>>> (device_particles_a, device_forces_a, num_particles, step_size, box_extension);
        apply_integrator_for_particle_euler <<< numberOfBlocks, numberOfThreads, 0, stream[1]>>> (device_particles_b, device_forces_b, num_particles, step_size, box_extension);
        cudaDeviceSynchronize();

        // Copy the particles back to the host
        cudaMemcpyAsync(host_particles_a, device_particles_a, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyDeviceToHost, stream[0]);
	    cudaMemcpyAsync(host_particles_b, device_particles_b, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyDeviceToHost, stream[1]);
        cudaDeviceSynchronize();

        // Write the VTK file
        writeVTKFile(step + 1, num_particles, host_particles_a, host_particles_b);
    }
}

// //particle simulation using graphs
// void start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity, float box_extension, float cut_off_radious)
// {
//     float3 *host_forces_a;
//     float3 *host_forces_b;
//     float3 *device_forces_a;
//     float3 *device_forces_b;

//     Particle3D *host_particles_a = NULL;
//     Particle3D *host_particles_b = NULL;
//     Particle3D* device_particles_a = NULL; 
//     Particle3D* device_particles_b = NULL; 

//     int deviceId;
//     cudaDeviceProp prop;

//     float num_cell_1d = box_extension / cut_off_radious;
//     float num_cell_total = powf(num_cell_1d, 3);

//     cudaGetDevice(&deviceId);
//     cudaGetDeviceProperties(&prop, deviceId);

//     int numberOfThreads = 256;
//     int numberOfBlocks = 32 * prop.multiProcessorCount;

//     cudaMallocHost((void**)&host_particles_a, sizeof(Particle3D) * (num_particles / 2));
//     cudaMallocHost((void**)&host_particles_b, sizeof(Particle3D) * (num_particles / 2));
//     cudaMallocHost((void**)&host_forces_a, sizeof(float3) * (num_particles / 2));
//     cudaMallocHost((void**)&host_forces_b, sizeof(float3) * (num_particles / 2));

//     cudaMalloc((void**)&device_particles_a, sizeof(Particle3D) * (num_particles / 2));
//     cudaMalloc((void**)&device_particles_b, sizeof(Particle3D) * (num_particles / 2));
//     cudaMalloc((void**)&device_forces_a, sizeof(float3) * (num_particles / 2));
//     cudaMalloc((void**)&device_forces_b, sizeof(float3) * (num_particles / 2));

//     cudaStream_t stream[2];
//     cudaGraph_t graph[2];
//     cudaGraphExec_t instance[2];

//     for (int i = 0; i < 2; i++) {
//         cudaStreamCreate(&stream[i]);
//     }

//     t_neighbourList *nb_list = nullptr;

//     int pos = 4;    

//     for (int i = 0; i < num_particles; ++i) {
//         float x = fmod(pos, box_extension) ;
//         float y = (pos >= box_extension) ? fmod(floor(pos * 2 / box_extension), box_extension): 0;
//         float z = (pos * 4 >= box_extension * box_extension) ? fmod(floor((pos * 4) / (box_extension * box_extension) ), box_extension) : 0;

//         if (i < (num_particles / 2)) {
//             host_particles_a[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
//             host_forces_a[i] = float3{ 0.0f, 0.0f, 0.0f };
//         } else {
//             host_particles_b[i - (num_particles / 2)] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f, 1.0f, nullptr, i);
//             host_forces_b[i - (num_particles / 2)] = float3{ 0.0f, 0.0f, 0.0f };
//         }
        
//         pos += 2;
//     }

//     std::cout << num_particles << ", ";

//     cudaMemcpyAsync(device_particles_a, host_particles_a, sizeof(Particle3D) * (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
//     cudaMemcpyAsync(device_particles_b, host_particles_b, sizeof(Particle3D) * (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);

//     cudaMemcpyAsync(device_forces_a, host_forces_a, sizeof(float3) * (num_particles / 2), cudaMemcpyHostToDevice, stream[0]);
//     cudaMemcpyAsync(device_forces_b, host_forces_b, sizeof(float3) * (num_particles / 2), cudaMemcpyHostToDevice, stream[1]);

//     // Write the particles initial state
//     writeVTKFile(0, num_particles, host_particles_a, host_particles_b);

//     for (int step = 0; step < time_steps; ++step) {
//         cudaMemset(device_forces_a, 0, sizeof(float3) * (num_particles / 2));
//         cudaMemset(device_forces_b, 0, sizeof(float3) * (num_particles / 2));

//         // Begin capture for stream 0
//         cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeGlobal);

//         compute_force_between_particles <<< numberOfBlocks, numberOfThreads, 0, stream[0]>>> (device_particles_a, device_forces_a, num_particles, eps, sigma, k_n, gamma, gravity, box_extension, cut_off_radious);
//         apply_integrator_for_particle_euler <<< numberOfBlocks, numberOfThreads, 0, stream[0]>>> (device_particles_a, device_forces_a, num_particles, step_size, box_extension);

//         // End capture for stream 0
//         cudaStreamEndCapture(stream[0], &graph[0]);
//         cudaGraphInstantiate(&instance[0], graph[0], NULL, NULL, 0);

//         // Begin capture for stream 1
//         cudaStreamBeginCapture(stream[1], cudaStreamCaptureModeGlobal);

//         compute_force_between_particles <<< numberOfBlocks, numberOfThreads, 0, stream[1]>>> (device_particles_b, device_forces_b, num_particles, eps, sigma, k_n, gamma, gravity, box_extension, cut_off_radious);
//         apply_integrator_for_particle_euler <<< numberOfBlocks, numberOfThreads, 0, stream[1]>>> (device_particles_b, device_forces_b, num_particles, step_size, box_extension);

//         // End capture for stream 1
//         cudaStreamEndCapture(stream[1], &graph[1]);
//         cudaGraphInstantiate(&instance[1], graph[1], NULL, NULL, 0);

//         for (int i = 0; i < 2; i++) {
//             cudaGraphLaunch(instance[i], stream[i]);
//         }

//         cudaDeviceSynchronize();

//         // Copy the particles back to the host
//         cudaMemcpyAsync(host_particles_a, device_particles_a, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyDeviceToHost, stream[0]);
// 	    cudaMemcpyAsync(host_particles_b, device_particles_b, sizeof(Particle3D)* (num_particles / 2), cudaMemcpyDeviceToHost, stream[1]);
//         cudaDeviceSynchronize();

//         writeVTKFile(step + 1, num_particles, host_particles_a, host_particles_b);

//         // Cleanup the graph instances after each iteration
//         for (int i = 0; i < 2; i++) {
//             cudaGraphDestroy(graph[i]);
//             cudaGraphExecDestroy(instance[i]);
//         }
//     }

//     cudaFree(device_particles_a);
//     cudaFree(device_particles_b);
//     cudaFree(device_forces_a);
//     cudaFree(device_forces_b);

//     cudaFreeHost(host_particles_a);
//     cudaFreeHost(host_particles_b);
//     cudaFreeHost(host_forces_a);
//     cudaFreeHost(host_forces_b);

//     for (int i = 0; i < 2; i++) {
//         cudaStreamDestroy(stream[i]);
//     }
// }

