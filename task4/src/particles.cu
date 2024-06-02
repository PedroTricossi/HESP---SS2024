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

__device__ int calculate_matrix_coordenates(int x, int y, int z, float num_cell_1d, float box_extension){
    x < 0 ? x = num_cell_1d - 1 : x;
    x >= num_cell_1d ? x = 0 : x;
    y < 0 ? y = num_cell_1d - 1 : y;
    y >= num_cell_1d ? y = 0 : y;
    z < 0 ? z = num_cell_1d - 1 : z;
    z >= num_cell_1d ? z = 0 : z;

    return fmod(x + y * num_cell_1d + z * num_cell_1d * num_cell_1d, num_cell_1d * num_cell_1d * num_cell_1d);
}


__device__ void Particle3D::get_neighbours(t_neighbourList *neighbourList, int *nb_list, float cut_off_radious, float box_extension){
    float num_cell_1d = box_extension / cut_off_radious;


    float x_pos = floor( position.x / cut_off_radious );
    float y_pos = floor( position.y / cut_off_radious );
    float z_pos = floor( position.z / cut_off_radious );
    
    int cell_index_0 = calculate_matrix_coordenates(x_pos, y_pos, z_pos, num_cell_1d, box_extension);
    int cell_index_1 = calculate_matrix_coordenates(x_pos + 1, y_pos, z_pos, num_cell_1d, box_extension);
    int cell_index_2 = calculate_matrix_coordenates(x_pos - 1, y_pos, z_pos, num_cell_1d, box_extension);
    int cell_index_3 = calculate_matrix_coordenates(x_pos, y_pos + 1, z_pos, num_cell_1d, box_extension);
    int cell_index_4 = calculate_matrix_coordenates(x_pos, y_pos - 1, z_pos, num_cell_1d, box_extension);
    int cell_index_5 = calculate_matrix_coordenates(x_pos, y_pos, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_6 = calculate_matrix_coordenates(x_pos, y_pos, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_7 = calculate_matrix_coordenates(x_pos + 1, y_pos + 1, z_pos, num_cell_1d, box_extension);
    int cell_index_8 = calculate_matrix_coordenates(x_pos - 1, y_pos - 1, z_pos, num_cell_1d, box_extension);
    int cell_index_9 = calculate_matrix_coordenates(x_pos + 1, y_pos - 1, z_pos, num_cell_1d, box_extension);
    int cell_index_10 = calculate_matrix_coordenates(x_pos - 1, y_pos + 1, z_pos, num_cell_1d, box_extension);
    int cell_index_11 = calculate_matrix_coordenates(x_pos + 1, y_pos, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_12 = calculate_matrix_coordenates(x_pos - 1, y_pos, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_13 = calculate_matrix_coordenates(x_pos - 1, y_pos , z_pos + 1, num_cell_1d, box_extension);
    int cell_index_14 = calculate_matrix_coordenates(x_pos + 1, y_pos , z_pos - 1, num_cell_1d, box_extension);
    int cell_index_15 = calculate_matrix_coordenates(x_pos, y_pos + 1, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_16 = calculate_matrix_coordenates(x_pos, y_pos - 1, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_17 = calculate_matrix_coordenates(x_pos, y_pos + 1, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_18 = calculate_matrix_coordenates(x_pos, y_pos - 1, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_19 = calculate_matrix_coordenates(x_pos + 1, y_pos + 1, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_20 = calculate_matrix_coordenates(x_pos - 1, y_pos - 1, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_21 = calculate_matrix_coordenates(x_pos + 1, y_pos - 1, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_22 = calculate_matrix_coordenates(x_pos - 1, y_pos + 1, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_23 = calculate_matrix_coordenates(x_pos - 1, y_pos - 1, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_24 = calculate_matrix_coordenates(x_pos + 1, y_pos + 1, z_pos - 1, num_cell_1d, box_extension);
    int cell_index_25 = calculate_matrix_coordenates(x_pos + 1, y_pos - 1, z_pos + 1, num_cell_1d, box_extension);
    int cell_index_26 = calculate_matrix_coordenates(x_pos - 1, y_pos + 1, z_pos + 1, num_cell_1d, box_extension);


    nb_list[0] = cell_index_0;
    nb_list[1] = cell_index_1;
    nb_list[2] = cell_index_2;
    nb_list[3] = cell_index_3;
    nb_list[4] = cell_index_4;
    nb_list[5] = cell_index_5;
    nb_list[6] = cell_index_6;
    nb_list[7] = cell_index_7;
    nb_list[8] = cell_index_8;
    nb_list[9] = cell_index_9;
    nb_list[10] = cell_index_10;    
    nb_list[11] = cell_index_11;
    nb_list[12] = cell_index_12;
    nb_list[13] = cell_index_13;
    nb_list[14] = cell_index_14;
    nb_list[15] = cell_index_15;
    nb_list[16] = cell_index_16;
    nb_list[17] = cell_index_17;
    nb_list[18] = cell_index_18;
    nb_list[19] = cell_index_19;
    nb_list[20] = cell_index_20;
    nb_list[21] = cell_index_21;
    nb_list[22] = cell_index_22;
    nb_list[23] = cell_index_23;
    nb_list[24] = cell_index_24;
    nb_list[25] = cell_index_25;
    nb_list[26] = cell_index_26;

}

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma, float box_extension, float cut_off_radious, t_neighbourList* nb_list) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float3 f;
    float3 r;
    int particle_nb[27];

    if (i < num_particles) {
        particles[i].get_neighbours(nb_list, particle_nb, cut_off_radious, box_extension);

        // printf("particle id: %d\n", particles[i].getId());
        for(int k = 0; k < 27; k++){
            int cell_index = particle_nb[k];
            t_neighbourList *current_cell = nb_list;

            for (int j = 0; j < cell_index; j++){
                current_cell = current_cell->next;
            }

            Particle3D *current_particle = current_cell->particle;

            // printf("current_particle: %d\n", current_particle->getId());
            

            while(current_particle != nullptr){
                if(current_particle->getId() != particles[i].getId()){
                    r.x = current_particle->getPosition().x - particles[i].getPosition().x;
                    r.y = current_particle->getPosition().y - particles[i].getPosition().y;
                    r.z = current_particle->getPosition().z - particles[i].getPosition().z;

                    float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

                    float force_ij = particles[i].forceUpdate(*current_particle, eps, sigma, box_extension);

                    f.x = force_ij * r.x / xij * xij;
                    f.y = force_ij * r.y / xij * xij;
                    f.z = force_ij * r.z / xij * xij;

                    // printf("force_ij: %f\n", f.x);


                    atomicAdd(&forces[i].x, -f.x);
                    atomicAdd(&forces[i].y, -f.y);
                    atomicAdd(&forces[i].z, -f.z);

                    

                    atomicAdd(&forces[current_particle->getId()].x, f.x);
                    atomicAdd(&forces[current_particle->getId()].y, f.y);
                    atomicAdd(&forces[current_particle->getId()].z, f.z);
                }   
                current_particle = current_particle->getNextParticle();
            }
            
        }
        
        // for (int j = 0; j < num_particles; ++j) {
        //     if (i != j) {
        //         r.x = particles[j].getPosition().x - particles[i].getPosition().x;
        //         r.y = particles[j].getPosition().y - particles[i].getPosition().y;
        //         r.z = particles[j].getPosition().z - particles[i].getPosition().z;

        //         float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        //         // if (xij <= cut_off_radious){
        //             float force_ij = particles[i].forceUpdate(particles[j], eps, sigma,box_extension);
                    
        //             f.x = force_ij * r.x / xij * xij;
        //             f.y = force_ij * r.y / xij * xij;
        //             f.z = force_ij * r.z / xij * xij;

        //             atomicAdd(&forces[i].x, -f.x);
        //             atomicAdd(&forces[i].y, -f.y);
        //             atomicAdd(&forces[i].z, -f.z);

        //             atomicAdd(&forces[j].x, f.x);
        //             atomicAdd(&forces[j].y, f.y);
        //             atomicAdd(&forces[j].z, f.z);
        //         // }
        //     }
        // }
    }

                
        
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
// }

__global__ void apply_integrator_for_particle(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        float3 acceleration;
        float3 half_speed;
        float3 new_position;
        float3 new_position_boundary;

        acceleration.x = forces[i].x / particle.getMass();
        acceleration.y = forces[i].y / particle.getMass();
        acceleration.z = forces[i].z / particle.getMass();

        // printf("acceleration: %f\n", acceleration.x);

        half_speed.x = particle.getVelocity().x + (0.5f * step_size * acceleration.x);
        half_speed.y = particle.getVelocity().y + (0.5f * step_size * acceleration.y);
        half_speed.z = particle.getVelocity().z + (0.5f * step_size * acceleration.z);
        

        new_position.x = particle.getPosition().x + (step_size * particle.getVelocity().x) + (step_size * step_size * acceleration.x * 0.5f);
        new_position.y = particle.getPosition().y + (step_size * particle.getVelocity().y) + (step_size * step_size * acceleration.y * 0.5f);
        new_position.z = particle.getPosition().z + (step_size * particle.getVelocity().z) + (step_size * step_size * acceleration.z * 0.5f);

        new_position_boundary.x = fmod(new_position.x, box_extension);
        new_position_boundary.y = fmod(new_position.y, box_extension);
        new_position_boundary.z = fmod(new_position.z, box_extension);

        // printf("new_position: %f, \n", new_position.x);

        if(new_position.x < 0)
            new_position.x = new_position_boundary.x + box_extension;
        
        else if(new_position.x >= (box_extension))
            new_position.x =  box_extension - new_position_boundary.x;
        
        if(new_position.y < 0)
            new_position.y = new_position_boundary.y + box_extension;
        
        else if(new_position.y >= (box_extension))
            new_position.y = box_extension - new_position_boundary.y;
        
        if(new_position.z < 0)
            new_position.z = new_position_boundary.z + box_extension;
        
        else if(new_position.z >= (box_extension))
            new_position.z = box_extension - new_position_boundary.z;     


        particle.setPosition(new_position);



        particle.setVelocity(float3{
                half_speed.x + (0.5f * step_size * acceleration.x),
                half_speed.y + (0.5f * step_size * acceleration.y),
                half_speed.z + (0.5f * step_size * acceleration.z),
        });
    }
}
