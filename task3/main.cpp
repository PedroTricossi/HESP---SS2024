#include <iostream>
#include "include/point.h"
#include "include/neighbourList.h"
#include "include/constants.h"
#include <vector> 
using namespace std; 

int main(int argc, char* argv[]) {
    int ret_value = 0;
    vector<t_point> particles(NUM_PARTICLES);
    t_neighbourList *neighbourList = init_neighbourList(SIMULATION_SPACE);
    int collision = 0;

    for(int i = 0; i < NUM_PARTICLES; i++){
        init_particle(&particles[i], &i);
        add_particle(neighbourList, &particles[i]);
    }

    for(int i = 0; i < NUM_INTERACTION; i++){
        cout << "Iteration: " << i << endl;

        for(t_point& particle : particles){
            particle = update(&particle, DT);
            ret_value = detect_collision(neighbourList, &particle);
            if(ret_value == 1){
                collision = 1;
            }
        }

        if(collision == 1){
            cout << "Collision detected" << endl;
            clean_particle(neighbourList);
            neighbourList = init_neighbourList(SIMULATION_SPACE);
            for(t_point& particle : particles){
                add_particle(neighbourList, &particle);
            }
            ret_value = 0;
            collision = 0;
        }

        t_neighbourList *current_cell = neighbourList;
        while(current_cell != NULL){
            if(current_cell->num_particles > 0)
                cout << "Cell " << current_cell->id << " has " << current_cell->num_particles << " particles" << endl;
            current_cell = current_cell->next;
        }
    }

 
    clean_particle(neighbourList);

    return 0;
}