// #ifndef particles.cuh
// #define  particles.cuh
#pragma once
#include <cstddef>
// #include "n_list.cuh"
typedef struct t_neighbourList;


class Particle3D
{
private:
    int id; // Unique identifier for the particle 
    Particle3D* next_particle; // Pointer to the next particle (Neighbour list)
    float3 position;  // Position of the particle
    float3 velocity;  // Velocity of the particle
    float mass;       // Mass of the particle
    float m_radius;     //Radius of the particle

public:
    // Constructors
    Particle3D()
            : mass(0.0f), m_radius(0.000005f) //Took 5 micrometers as radius
    {
        position.x = 0.0f;
        position.y = 0.0f;
        position.z = 0.0f;

        velocity.x = 0.0f;
        velocity.y = 0.0f;
        velocity.z = 0.0f;

        next_particle = nullptr ;
    }

    Particle3D(float3 pos, float3 vel, float m,float radius, Particle3D* np, int id)
            : position(pos), velocity(vel), mass(m), m_radius(radius), next_particle(np), id(id)
    {
    }

    

    // Getters
    __host__ __device__ int getId() const { return id; }
    __host__ __device__ float3 getPosition() const { return position; }
    __host__ __device__ float3 getVelocity() const { return velocity; }
    __host__ __device__ float getMass() const { return mass; }
    __host__ __device__ Particle3D* getNextParticle() const { return next_particle; }
    __host__ __device__ float getRadius() const { return m_radius; }

    // Setters
    __host__ __device__ void setId(int i) { id = i; }
    __host__ __device__ void setPosition(float3 pos) { position = pos; }
    __host__ __device__ void setVelocity(float3 vel) { velocity = vel; }
    __host__ __device__ void setMass(float m) { mass = m; }
    __host__ __device__ void setNextParticle(Particle3D* np) { next_particle = np; }
    __host__ __device__ void setRadius(float r) { m_radius = r; }


    // Function to calculate the force update
    __host__ __device__ float forceUpdate(const Particle3D& particle_j, const float eps, const float sigma, float box_extension);
    __host__ __device__ void get_neighbours(t_neighbourList *neighbourList, int *nb_list, float cut_off_radious, float box_extension);
    __host__ __device__ float3 calculate_spring_dashpot_force(const Particle3D& particle_j, float k_n, float gamma, float box_extension);
    
};

__host__ __device__ int calculate_matrix_coordenates(int x, int y, int z, float num_cell_1d, float box_extension);

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity, 
float box_extension, float cut_off_radious, t_neighbourList* nb_list);

__global__ void apply_integrator_for_particle(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension);
__global__ void apply_integrator_for_particle_euler(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension);
__global__ void apply_integrator_for_particle_rk4(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension);

// #endif 
