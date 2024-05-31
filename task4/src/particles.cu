#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "../include/particles.cuh"
#include "../include/n_list.cuh"

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

__device__ void Particle3D::get_neighbours(t_neighbourList *neighbourList, t_neighbourList * nb_list, float cut_off_radious, float box_extension){
    float num_cell_1d = box_extension / cut_off_radious;

    int x_pos = int(position.x / cut_off_radious);
    int y_pos = int(position.y / cut_off_radious) * num_cell_1d;
    int z_pos = int(position.z / cut_off_radious) * num_cell_1d * num_cell_1d;
    
    int cell_index_0 = x_pos + y_pos + z_pos;
    int cell_index_1 = x_pos + y_pos + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_2 = x_pos + y_pos + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_3 = x_pos + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + z_pos;
    int cell_index_4 = x_pos + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + z_pos;
    int cell_index_5 = x_pos + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_6 = x_pos + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_7 = x_pos + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_8 = x_pos + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_9 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + y_pos + z_pos;
    int cell_index_10 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + y_pos + z_pos;
    int cell_index_11 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + y_pos + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_12 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + y_pos + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_13 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + y_pos + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_14 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + y_pos + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_15 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + z_pos;
    int cell_index_16 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + z_pos;
    int cell_index_17 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + z_pos;
    int cell_index_18 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + z_pos;
    int cell_index_19 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_20 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_21 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_22 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + (y_pos + 1 > num_cell_1d ? 0 : y_pos + 1 ) + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_23 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_24 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + (z_pos + 1 > num_cell_1d ? 0 : z_pos + 1);
    int cell_index_25 = (x_pos + 1 > num_cell_1d ? 0 : x_pos + 1 ) + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);
    int cell_index_26 = (x_pos - 1 < 0 ? num_cell_1d : x_pos - 1 ) + (y_pos - 1 < 0 ? num_cell_1d : y_pos - 1 ) + (z_pos - 1 < 0 ? num_cell_1d : z_pos - 1);


    nb_list[0] = neighbourList[cell_index_0];
    nb_list[1] = neighbourList[cell_index_1];
    nb_list[2] = neighbourList[cell_index_2];
    nb_list[3] = neighbourList[cell_index_3];
    nb_list[4] = neighbourList[cell_index_4];
    nb_list[5] = neighbourList[cell_index_5];
    nb_list[6] = neighbourList[cell_index_6];
    nb_list[7] = neighbourList[cell_index_7];
    nb_list[8] = neighbourList[cell_index_8];
    nb_list[9] = neighbourList[cell_index_9];
    nb_list[10] = neighbourList[cell_index_10];
    nb_list[11] = neighbourList[cell_index_11];
    nb_list[12] = neighbourList[cell_index_12];
    nb_list[13] = neighbourList[cell_index_13];
    nb_list[14] = neighbourList[cell_index_14];
    nb_list[15] = neighbourList[cell_index_15];
    nb_list[16] = neighbourList[cell_index_16];
    nb_list[17] = neighbourList[cell_index_17];
    nb_list[18] = neighbourList[cell_index_18];
    nb_list[19] = neighbourList[cell_index_19];
    nb_list[20] = neighbourList[cell_index_20];
    nb_list[21] = neighbourList[cell_index_21];
    nb_list[22] = neighbourList[cell_index_22];
    nb_list[23] = neighbourList[cell_index_23];
    nb_list[24] = neighbourList[cell_index_24];
    nb_list[25] = neighbourList[cell_index_25];
    nb_list[26] = neighbourList[cell_index_26];
}

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma, float box_extension, float cut_off_radious, t_neighbourList* nb_list) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 f;
    float3 r;
    t_neighbourList particle_nb[27];


    if (i < num_particles) {
        particles[i].get_neighbours(nb_list, particle_nb, cut_off_radious, box_extension);

        for(int k = 0; k < 27; k++){
            if(particle_nb[k].num_particles > 0){
                Particle3D* particles_in_cell = particle_nb[k].particle;
                while(particles_in_cell != nullptr){
                    if(particles_in_cell->getId() != particles[i].getId()){
                        float force_ij = particles[i].forceUpdate(*particles_in_cell, eps, sigma, box_extension);

                        printf("force_ij: %f\n", force_ij);

                        r.x = particles_in_cell->getPosition().x - particles[i].getPosition().x;
                        r.y = particles_in_cell->getPosition().y - particles[i].getPosition().y;
                        r.z = particles_in_cell->getPosition().z - particles[i].getPosition().z;

                        float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

                        f.x = force_ij * r.x / xij * xij;
                        f.y = force_ij * r.y / xij * xij;
                        f.z = force_ij * r.z / xij * xij;

                        atomicAdd(&forces[i].x, -f.x);
                        atomicAdd(&forces[i].y, -f.y);
                        atomicAdd(&forces[i].z, -f.z);

                        atomicAdd(&forces[k].x, f.x);
                        atomicAdd(&forces[k].y, f.y);
                        atomicAdd(&forces[k].z, f.z);
                    }
                    particles_in_cell = particles_in_cell->getNextParticle();
                    // printf("particles_in_cell: %p\n", (void *) particles_in_cell);
                }
                // printf("out of while\n");
            }
                

                
                // if (xij <= cut_off_radious){
                //     float force_ij = particles[i].forceUpdate(particles[j], eps, sigma, box_extension);

                //     f.x = force_ij * r.x / xij * xij;
                //     f.y = force_ij * r.y / xij * xij;
                //     f.z = force_ij * r.z / xij * xij;

                //     atomicAdd(&forces[i].x, -f.x);
                //     atomicAdd(&forces[i].y, -f.y);
                //     atomicAdd(&forces[i].z, -f.z);

                //     atomicAdd(&forces[j].x, f.x);
                //     atomicAdd(&forces[j].y, f.y);
                //     atomicAdd(&forces[j].z, f.z);
                // }
            
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