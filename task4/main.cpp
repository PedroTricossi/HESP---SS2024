#include <iostream>


int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " time_steps step_size num_particles eps sigma" << std::endl;
        return 1;
    }

    int time_steps = std::atoi(argv[1]);
    float step_size = std::atof(argv[2]);
    int num_particles = std::atoi(argv[3]);
    float eps = std::atof(argv[4]);
    float sigma = std::atof(argv[5]);

    return 0;
}
