#pragma once
#include <cstddef>

class Particle3D
{
private:
    Particle3D* next_particle; // Pointer to the next particle (Neighbour list)
    float3 position;  // Position of the particle
    float3 velocity;  // Velocity of the particle
    float mass;       // Mass of the particle
    
    

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

        next_particle = nullptr ;
    }

    Particle3D(float3 pos, float3 vel, float m, Particle3D* np)
            : position(pos), velocity(vel), mass(m), next_particle(np)
    {
    }

    

    // Getters
    __host__ __device__ float3 getPosition() const { return position; }
    __host__ __device__ float3 getVelocity() const { return velocity; }
    __host__ __device__ float getMass() const { return mass; }
    __host__ __device__ Particle3D* getNextParticle() const { return next_particle; }

    // Setters
    __host__ __device__ void setPosition(float3 pos) { position = pos; }
    __host__ __device__ void setVelocity(float3 vel) { velocity = vel; }
    __host__ __device__ void setMass(float m) { mass = m; }
    __host__ __device__ void setNextParticle(Particle3D* np) { next_particle = np; }

    // Function to calculate the force update
    __host__ __device__ float forceUpdate(const Particle3D& particle_j, const float eps, const float sigma, float box_extension);
};

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma, float box_extension, float cut_off_radious);

__global__ void apply_integrator_for_particle(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension);