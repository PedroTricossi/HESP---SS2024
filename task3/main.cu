#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Particle3D
{
private:
    float3 position; // Position of the particle
    float3 velocity; // Velocity of the particle
    float mass;      // Mass of the particle

public:
    // Constructors
    Particle3D()
        : mass(0.0f)
    {
        position.x = 0.0f;
        position.y = 0.0f;
        position.z = 0.0f;
        velocity.x = 0.0f;
        velocity.y = 0.0f;
        velocity.z = 0.0f;
    }

    Particle3D(float3 pos, float3 vel, float m)
        : position(pos), velocity(vel), mass(m)
    {
    }

    // Getters
    __host__ __device__ float3 getPosition() const { return position; }
    __host__ __device__ float3 getVelocity() const { return velocity; }
    __host__ __device__ float getMass() const { return mass; }

    // Setters
    __host__ __device__ void setPosition(float3 pos) { position = pos; }
    __host__ __device__ void setVelocity(float3 vel) { velocity = vel; }
    __host__ __device__ void setMass(float m) { mass = m; }

    // Function to calculate the LJ potential between two particles
    __host__ __device__ float calculateLJPotential(const Particle3D& particle, const float eps, const float sigma);

    // Function to calculate the force update
    __host__ __device__ float forceUpdate(const Particle3D& particle, const float eps, const float sigma);
};

__device__ float Particle3D::calculateLJPotential(const Particle3D& particle, const float eps, const float sigma)
{
    float3 r;
    r.x = particle.getPosition().x - position.x;
    r.y = particle.getPosition().y - position.y;
    r.z = particle.getPosition().z - position.z;

    float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    float sigma_xij = sigma / xij;
    return (4 * eps * (powf(sigma_xij, 12) - powf(sigma_xij, 6)));
}

__device__ float Particle3D::forceUpdate(const Particle3D& particle, const float eps, const float sigma)
{
    float3 r;
    r.x = particle.getPosition().x - position.x;
    r.y = particle.getPosition().y - position.y;
    r.z = particle.getPosition().z - position.z;

    float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    float sigma_xij = sigma / xij;

    float f_scalar = 24 * eps * powf(sigma_xij, 6) * 2 * (powf(sigma_xij, 6) - 1) * (xij / (xij * xij));

    return f_scalar;
}


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

__global__ void computeForces(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        for (int j = 0; j < num_particles; ++j) {
            if (i != j) {
                float force_ij = particles[i].forceUpdate(particles[j], eps, sigma);
                float3 r;
                r.x = particles[j].getPosition().x - particles[i].getPosition().x;
                r.y = particles[j].getPosition().y - particles[i].getPosition().y;
                r.z = particles[j].getPosition().z - particles[i].getPosition().z;
                float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
                float3 f;
                f.x = force_ij * r.x / xij;
                f.y = force_ij * r.y / xij;
                f.z = force_ij * r.z / xij;
                atomicAdd(&forces[i].x, -f.x);
                atomicAdd(&forces[i].y, -f.y);
                atomicAdd(&forces[i].z, -f.z);
                atomicAdd(&forces[j].x, f.x);
                atomicAdd(&forces[j].y, f.y);
                atomicAdd(&forces[j].z, f.z);
            }
        }
    }
}

__global__ void integrateParticles(Particle3D* particles, float3* forces, int num_particles, float step_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        float3 acceleration;
        acceleration.x = forces[i].x / particle.getMass();
        acceleration.y = forces[i].y / particle.getMass();
        acceleration.z = forces[i].z / particle.getMass();
        particle.setPosition(float3{
            particle.getPosition().x + (step_size * particle.getVelocity().x) + (step_size * step_size * acceleration.x * 0.5f),
            particle.getPosition().y + (step_size * particle.getVelocity().y) + (step_size * step_size * acceleration.y * 0.5f),
            particle.getPosition().z + (step_size * particle.getVelocity().z) + (step_size * step_size * acceleration.z * 0.5f)
            });
        particle.setVelocity(float3{
            particle.getVelocity().x + (0.5f * step_size * acceleration.x),
            particle.getVelocity().y + (0.5f * step_size * acceleration.y),
            particle.getVelocity().z + (0.5f * step_size * acceleration.z)
            });
    }
}

void simulateParticles(Particle3D* particles, float3* forces, int time_steps, int num_particles, float eps, float sigma, float step_size) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0

    int numberOfThreads = 256; // Declare and initialize numberOfThreads
    int numberOfBlocks = 32 * prop.multiProcessorCount; // Declare and initialize numberOfBlocks

    for (int step = 0; step < time_steps; ++step) {
        // Reset forces
        cudaMemset(forces, 0, num_particles * sizeof(float3));

        // Compute forces using CUDA
        computeForces << <numberOfBlocks, numberOfThreads >> > (particles, forces, num_particles, eps, sigma);
        cudaDeviceSynchronize();

        // Integrate particles using CUDA
        integrateParticles << <numberOfBlocks, numberOfThreads >> > (particles, forces, num_particles, step_size);
        cudaDeviceSynchronize();

        // Write the VTK file
        writeVTKFile(step + 1, num_particles, particles);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " time_steps step_size num_particles eps sigma" << std::endl;
        return 1;
    }

    int time_steps = std::atoi(argv[1]);
    float step_size = std::atof(argv[2]);
    int num_particles = std::atoi(argv[3]);
    float eps = std::atof(argv[4]);
    float sigma = std::atof(argv[5]);

    Particle3D* particles;
    float3* forces;

    cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    cudaMallocManaged(&forces, num_particles * sizeof(float3));

    for (int i = 0; i < num_particles; ++i) {
        float x = (i == 0) ? i + 1 : i + 3;
        float y = x;
        float z = x;
        particles[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f);
        forces[i] = float3{ 0.0f, 0.0f, 0.0f };
    }

    writeVTKFile(0, num_particles, particles);

    simulateParticles(particles, forces, time_steps, num_particles, eps, sigma, step_size);

    cudaFree(particles);
    cudaFree(forces);

    return 0;
}
