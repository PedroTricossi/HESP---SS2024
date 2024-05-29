#include "include/particles.cuh"

void start_particle_simulation(){
    Particle3D* particles;
    float3* forces;

    cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    cudaMallocManaged(&forces, num_particles * sizeof(float3));

    for (int i = 0; i < num_particles; ++i) {
        float x = fmod(i, 10);
        float y = (i >= 10) ? fmod(floor(i / 10), 10): 0;
        float z = (i >= 100) ? fmod(floor(i / 100), 10) : 0;
        particles[i] = Particle3D(float3{ x, y, z }, float3{ 0.0f, 0.0f, 0.0f }, 1.0f);
        forces[i] = float3{ 0.0f, 0.0f, 0.0f };
    }

    writeVTKFile(0, num_particles, particles);

    simulateParticles(particles, forces, time_steps, num_particles, eps, sigma, step_size);

    cudaFree(particles);
    cudaFree(forces);
}