#include "include/particles.cuh"

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 f;
    float3 r;

    if (i < num_particles) {
        for (int j = 0; j < num_particles; ++j) {
            if (i != j) {
                float force_ij = particles[i].forceUpdate(particles[j], eps, sigma);

                r.x = particles[j].getPosition().x - particles[i].getPosition().x;
                r.y = particles[j].getPosition().y - particles[i].getPosition().y;
                r.z = particles[j].getPosition().z - particles[i].getPosition().z;

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

__global__ void apply_integrator_for_particle(Particle3D* particles, float3* forces, int num_particles, float step_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        float3 acceleration;
        float3 half_speed;
        acceleration.x = forces[i].x / particle.getMass();
        acceleration.y = forces[i].y / particle.getMass();
        acceleration.z = forces[i].z / particle.getMass();

        particle.setPosition(float3{
                particle.getPosition().x + (step_size * particle.getVelocity().x) + (step_size * step_size * acceleration.x * 0.5f),
                particle.getPosition().y + (step_size * particle.getVelocity().y) + (step_size * step_size * acceleration.y * 0.5f),
                particle.getPosition().z + (step_size * particle.getVelocity().z) + (step_size * step_size * acceleration.z * 0.5f)
        });

        half_speed.x = particle.getVelocity().x + (0.5f * step_size * acceleration.x);
        half_speed.y = particle.getVelocity().y + (0.5f * step_size * acceleration.y);
        half_speed.z = particle.getVelocity().z + (0.5f * step_size * acceleration.z);

        particle.setVelocity(float3{
                half_speed.x + (0.5f * step_size * acceleration.x),
                half_speed.y + (0.5f * step_size * acceleration.y),
                half_speed.z + (0.5f * step_size * acceleration.z),
        });
    }
}