#include <iostream>
#include "include/point.h"

int main(int argc, char* argv[]) {

    t_point particle;
    init_particle(&particle);
    for(int i = 0; i < 10; i++){
        update(&particle, 1/10.);

        std::cout << "Particle position: " << particle.cur_pos.x << ", " << particle.cur_pos.y << ", " << particle.cur_pos.z << std::endl;
        std::cout << "Particle velocity: " << particle.cur_vel.x << ", " << particle.cur_vel.y << ", " << particle.cur_vel.z << std::endl;
    }


    return 0;
}