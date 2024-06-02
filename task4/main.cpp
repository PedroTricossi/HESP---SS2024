#include <iostream>
#include <chrono>
#include <cmath>
#include "include/particle_simulation.cuh"


int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " time_steps step_size num_particles eps sigma box_extension cut_off_radious" << std::endl;
        return 1;
    }

    int time_steps = std::atoi(argv[1]);
    float step_size = std::atof(argv[2]);
    int num_particles = std::atoi(argv[3]);
    float eps = std::atof(argv[4]);
    float sigma = std::atof(argv[5]);
    float box_extension = std::atof(argv[6]);
    float cut_off_radious = std::atof(argv[7]);

    if(fmod(box_extension, cut_off_radious) != 0){
        std::cerr << "The extension of the boundery MUST be a multiple of the cut-off Radious" << std::endl;
        return 1;
    }

    std::cout << "BOX: " << box_extension << std::endl;

    auto start = std::chrono::steady_clock::now();

    start_particle_simulation(time_steps, step_size, num_particles, eps, sigma, box_extension, cut_off_radious);

    auto end = std::chrono::steady_clock::now();

    printf("Time: %f\n", std::chrono::duration<double>(end - start).count());

    return 0;
}
