#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "particles.cuh"
// #include "include/n_list.cuh"

__device__ float Particle3D::forceUpdate(const Particle3D& particle_j, const float eps, const float sigma, float box_extension)
{
        float3 r;
        float dx;
        float dy;
        float dz;

        dx = particle_j.getPosition().x - position.x;
        dy = particle_j.getPosition().y - position.y;
        dz = particle_j.getPosition().z - position.z;

        // https://en.wikipedia.org/wiki/Periodic_boundary_conditions

        r.x = (dx) - int(dx / box_extension) * box_extension;
        r.y = (dy) - int(dy / box_extension) * box_extension;
        r.z = (dz) - int(dz / box_extension) * box_extension;

        float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        float sigma_xij = sigma / xij;
        float sigma_xij_6 = powf(sigma_xij, 6);

        float f_scalar = 24 * eps * sigma_xij_6 * ((2 * sigma_xij_6) - 1) / (xij * xij);

        return f_scalar;
}

// Spring-dashpot force calculation function
__device__ float3 Particle3D::calculate_spring_dashpot_force(const Particle3D& particle_j, float k_n, float gamma, float box_extension) {
    // Relative position vector
    float3 r;
    r.x = particle_j.getPosition().x - position.x;
    r.y = particle_j.getPosition().y - position.y;
    r.z = particle_j.getPosition().z - position.z;

    // Apply periodic boundary conditions
    r.x -= round(r.x / box_extension) * box_extension;
    r.y -= round(r.y / box_extension) * box_extension;
    r.z -= round(r.z / box_extension) * box_extension;

    float distance = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

    // printf("distance: %f\n", distance);

    float overlap = distance - (m_radius + particle_j.getRadius());

    // printf("overlap: %f\n", overlap);

    // If there is no overlap, return zero force
    if (overlap > 0.001f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // printf("Overlap\n");

    // Unit normal vector
    float3 n;
    n.x = r.x / distance;
    n.y = r.y / distance;
    n.z = r.z / distance;

    // printf("n: %f\n", n.y);

    // Elastic force
    float3 f_elastic;
    f_elastic.x = k_n * overlap * n.x;
    f_elastic.y = k_n * overlap * n.y;
    f_elastic.z = k_n * overlap * n.z;

    // printf("k_n: %f\n", k_n);
    // printf("f_elastic: %f\n", f_elastic.y);

    // Relative velocity
    float3 v_rel;
    v_rel.x = particle_j.getVelocity().x - velocity.x;
    v_rel.y = particle_j.getVelocity().y - velocity.y;
    v_rel.z = particle_j.getVelocity().z - velocity.z;

    // printf("v_rel: %f\n", v_rel.y);

    // Damping force
    float3 f_damping;
    float dot_product = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;
    f_damping.x = gamma * dot_product * n.x;
    f_damping.y = gamma * dot_product * n.y;
    f_damping.z = gamma * dot_product * n.z;

    // printf("f_damping: %f\n", f_damping.y);

    // Total force
    float3 f_total;
    f_total.x = f_elastic.x + f_damping.x;
    f_total.y = f_elastic.y + f_damping.y;
    f_total.z = f_elastic.z + f_damping.z;

    // printf("f_total: %f\n", f_total.y);

    return f_total;
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

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity, 
float box_extension, float cut_off_radious, t_neighbourList* nb_list) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float3 f;
    float3 r;
    int particle_nb[27];

    if (i < num_particles) {
        // Gravitational force
        float3 gravitational_force;
        gravitational_force.x = 0.0f;
        gravitational_force.y = -gravity * particles[i].getMass();  // Assuming gravity acts along -y direction
        gravitational_force.z = 0.0f;

        // Apply gravitational force to particle i

        forces[i].x = forces[i].x + gravitational_force.x;
        forces[i].y = forces[i].y + gravitational_force.y;
        forces[i].z = forces[i].z + gravitational_force.z;

        

        // Loop over all particles to calculate the force between particle i and all other particles
        for (int j = 0; j < num_particles; ++j) {
            if (i != j) {
                r.x = particles[j].getPosition().x - particles[i].getPosition().x;
                r.y = particles[j].getPosition().y - particles[i].getPosition().y;
                r.z = particles[j].getPosition().z - particles[i].getPosition().z;

                float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
                if (xij <= cut_off_radious){
                    float3 force_ij = particles[i].calculate_spring_dashpot_force(particles[j], k_n, gamma, box_extension);

                    // printf("force_ij: %f\n", force_ij.y);
                    
                    // f.x = force_ij.x * r.x;
                    // f.y = force_ij.y * r.y;
                    // f.z = force_ij.z * r.z;

                    // printf("f: %f\n", f.y);

                    // printf("gravitational_force: %f\n", forces[i].y);

                    forces[i].x = forces[i].x + force_ij.x;
                    forces[i].y = forces[i].y + force_ij.y;
                    forces[i].z = forces[i].z + force_ij.z;

                    // printf("forces: %f\n", forces[i].y);
                }
            }
        }

    }
}

/*
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

        

        if(new_position.x < 0)
            new_position.x = box_extension - new_position_boundary.x;
        
        else if(new_position.x >= (box_extension)){
            new_position.x = new_position_boundary.x;
        }
        
        if(new_position.y < 0)
            new_position.y = new_position_boundary.y + box_extension;
        
        else if(new_position.y >= (box_extension))
            new_position.y = new_position_boundary.y;
        
        if(new_position.z < 0)
            new_position.z = new_position_boundary.z + box_extension;
        
        else if(new_position.z >= (box_extension))
            new_position.z = new_position_boundary.z;     
        
        // printf("new_position: %f, \n", new_position.x);

        particle.setPosition(new_position);



        particle.setVelocity(float3{
                half_speed.x + (0.5f * step_size * acceleration.x),
                half_speed.y + (0.5f * step_size * acceleration.y),
                half_speed.z + (0.5f * step_size * acceleration.z),
        });
    }
*/

//Explicit Euler
__global__ void apply_integrator_for_particle_euler(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        float3 acceleration;
        float3 new_velocity;
        float3 new_position;

        acceleration.x = forces[i].x / particle.getMass();
        acceleration.y = forces[i].y / particle.getMass();
        acceleration.z = forces[i].z / particle.getMass();

        new_velocity.x = particle.getVelocity().x + step_size * acceleration.x;
        new_velocity.y = particle.getVelocity().y + step_size * acceleration.y;
        new_velocity.z = particle.getVelocity().z + step_size * acceleration.z;

        new_position.x = particle.getPosition().x + step_size * new_velocity.x;
        new_position.y = particle.getPosition().y + step_size * new_velocity.y;
        new_position.z = particle.getPosition().z + step_size * new_velocity.z;

        new_position.x = fmodf(new_position.x + box_extension, box_extension);
        new_position.y = fmodf(new_position.y + box_extension, box_extension);
        new_position.z = fmodf(new_position.z + box_extension, box_extension);

        particle.setPosition(new_position);
        particle.setVelocity(new_velocity);
    }
}

// Runge Kutta
__global__ void apply_integrator_for_particle_rk4(Particle3D* particles, float3* forces, int num_particles, float step_size, float box_extension) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        float3 k1_pos, k1_vel;
        float3 k2_pos, k2_vel;
        float3 k3_pos, k3_vel;
        float3 k4_pos, k4_vel;
        float3 pos, vel, acc;

        pos = particle.getPosition();
        vel = particle.getVelocity();
        acc.x = forces[i].x / particle.getMass();
        acc.y = forces[i].y / particle.getMass();
        acc.z = forces[i].z / particle.getMass();

        // k1
        k1_vel = acc;
        k1_pos = vel;

        // k2
        k2_vel.x = forces[i].x / particle.getMass();
        k2_vel.y = forces[i].y / particle.getMass();
        k2_vel.z = forces[i].z / particle.getMass();
        k2_pos.x = vel.x + 0.5f * step_size * k1_vel.x;
        k2_pos.y = vel.y + 0.5f * step_size * k1_vel.y;
        k2_pos.z = vel.z + 0.5f * step_size * k1_vel.z;

        // k3
        k3_vel.x = forces[i].x / particle.getMass();
        k3_vel.y = forces[i].y / particle.getMass();
        k3_vel.z = forces[i].z / particle.getMass();
        k3_pos.x = vel.x + 0.5f * step_size * k2_vel.x;
        k3_pos.y = vel.y + 0.5f * step_size * k2_vel.y;
        k3_pos.z = vel.z + 0.5f * step_size * k2_vel.z;

        // k4
        k4_vel.x = forces[i].x / particle.getMass();
        k4_vel.y = forces[i].y / particle.getMass();
        k4_vel.z = forces[i].z / particle.getMass();
        k4_pos.x = vel.x + step_size * k3_vel.x;
        k4_pos.y = vel.y + step_size * k3_vel.y;
        k4_pos.z = vel.z + step_size * k3_vel.z;

        // Update position and velocity
        pos.x = fmodf(pos.x + (step_size / 6.0f) * (k1_pos.x + 2.0f * k2_pos.x + 2.0f * k3_pos.x + k4_pos.x) + box_extension, box_extension);
        pos.y = fmodf(pos.y + (step_size / 6.0f) * (k1_pos.y + 2.0f * k2_pos.y + 2.0f * k3_pos.y + k4_pos.y) + box_extension, box_extension);
        pos.z = fmodf(pos.z + (step_size / 6.0f) * (k1_pos.z + 2.0f * k2_pos.z + 2.0f * k3_pos.z + k4_pos.z) + box_extension, box_extension);

        vel.x += (step_size / 6.0f) * (k1_vel.x + 2.0f * k2_vel.x + 2.0f * k3_vel.x + k4_vel.x);
        vel.y += (step_size / 6.0f) * (k1_vel.y + 2.0f * k2_vel.y + 2.0f * k3_vel.y + k4_vel.y);
        vel.z += (step_size / 6.0f) * (k1_vel.z + 2.0f * k2_vel.z + 2.0f * k3_vel.z + k4_vel.z);

        particle.setPosition(pos);
        particle.setVelocity(vel);
    }
}

// }
