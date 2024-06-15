#include <iostream>
#include <cstdlib>
#include <chrono>
#include "particle_simulation.h" // Assuming this includes your CUDA kernel function

int main(int argc, char* argv[]) 
{
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0] << " time_steps step_size num_particles eps sigma box_extension cut_off_radious k_n gamma gravity" << std::endl;
        return 1;
    }

    int time_steps = std::atoi(argv[1]);
    float step_size = std::atof(argv[2]);
    int num_particles = std::atoi(argv[3]);
    float eps = std::atof(argv[4]);
    float sigma = std::atof(argv[5]);
    float box_extension = std::atof(argv[6]);
    float cut_off_radious = std::atof(argv[7]);
    float k_n = std::atof(argv[8]);
    float gamma = std::atof(argv[9]);
    float gamma = std::atof(argv[10]);

    if (fmod(box_extension, cut_off_radious) != 0) {
        std::cerr << "The extension of the boundary MUST be a multiple of the cut-off Radius" << std::endl;
        return 1;
    }

    std::cout << "BOX: " << box_extension << std::endl;

    auto start = std::chrono::steady_clock::now();

    start_particle_simulation(int time_steps, float step_size, int num_particles, float eps, float sigma, float k_n, float gamma, float gravity,float box_extension, float cut_off_radious);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

    return 0;
}
