#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/particles.cuh"

__device__ float Particle3D::forceUpdate(const Particle3D& particle_j, const float eps, const float sigma, float box_extension)
{
        float3 r;
        float dx;
        float dy;
        float dz;

        dx = particle_j.getPosition().x - position.x;
        dy = particle_j.getPosition().y - position.y;
        dz = particle_j.getPosition().z - position.z;

        r.x = (dx) - int(dx / box_extension) * box_extension;
        r.y = (dy) - int(dy / box_extension) * box_extension;
        r.z = (dz) - int(dz / box_extension) * box_extension;

        float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        float sigma_xij = sigma / xij;
        float sigma_xij_6 = powf(sigma_xij, 6);

        float f_scalar = 24 * eps * sigma_xij_6 * ( 2 * powf(sigma_xij, 6) - 1);

        return f_scalar;
}

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma, float box_extension) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 f;
    float3 r;

    if (i < num_particles) {
        for (int j = 0; j < num_particles; ++j) {
            if (i != j) {
                float force_ij = particles[i].forceUpdate(particles[j], eps, sigma, box_extension);
                float dx;
                float dy;
                float dz;


                dx = particles[j].getPosition().x - particles[i].getPosition().x;
                dy = particles[j].getPosition().y - particles[i].getPosition().y;
                dz = particles[j].getPosition().z - particles[i].getPosition().z;

                r.x = (dx) - int(dx / box_extension) * box_extension;
                r.y = (dy) - int(dy / box_extension) * box_extension;
                r.z = (dz) - int(dz / box_extension) * box_extension;

                float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

                f.x = force_ij * r.x / xij * xij;
                f.y = force_ij * r.y / xij * xij;
                f.z = force_ij * r.z / xij * xij;

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

__global__ void apply_integrator_for_particle(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        float3 acceleration;
        float3 half_speed;
        float3 new_position;

        acceleration.x = forces[i].x / particle.getMass();
        acceleration.y = forces[i].y / particle.getMass();
        acceleration.z = forces[i].z / particle.getMass();

        half_speed.x = particle.getVelocity().x + (0.5f * step_size * acceleration.x);
        half_speed.y = particle.getVelocity().y + (0.5f * step_size * acceleration.y);
        half_speed.z = particle.getVelocity().z + (0.5f * step_size * acceleration.z);
        

        new_position.x = particle.getPosition().x + (step_size * particle.getVelocity().x) + (step_size * step_size * acceleration.x * 0.5f);
        new_position.y = particle.getPosition().y + (step_size * particle.getVelocity().y) + (step_size * step_size * acceleration.y * 0.5f);
        new_position.z = particle.getPosition().z + (step_size * particle.getVelocity().z) + (step_size * step_size * acceleration.z * 0.5f);

        if(new_position.x < -(box_extension * 0.5f))
            new_position.x = fmod(new_position.x + box_extension, (box_extension * 0.5f));
        
        else if(new_position.x >= (box_extension * 0.5f))
            new_position.x = fmod(new_position.x - box_extension, -(box_extension * 0.5f));
        
        if(new_position.y < -(box_extension * 0.5f))
            new_position.y = fmod(new_position.y + box_extension, (box_extension * 0.5f));
        
        else if(new_position.y >= (box_extension * 0.5f))
            new_position.y = fmod(new_position.y - box_extension, -(box_extension * 0.5f));
        
        if(new_position.z < -(box_extension * 0.5f))
            new_position.z = fmod(new_position.z + box_extension, (box_extension * 0.5f));
        
        else if(new_position.z >= (box_extension * 0.5f))
            new_position.z = fmod(new_position.z - box_extension, -(box_extension * 0.5f));
        


        particle.setPosition(new_position);



        particle.setVelocity(float3{
                half_speed.x + (0.5f * step_size * acceleration.x),
                half_speed.y + (0.5f * step_size * acceleration.y),
                half_speed.z + (0.5f * step_size * acceleration.z),
        });
    }
}