#include <stdlib.h>
#include <math.h>
#include "../include/neighbourList.h"
#include "../include/constants.h"


t_neighbourList* init_neighbourList(int size){
    int matrix_size = size * size * size;
    t_neighbourList *neighbourList = (t_neighbourList*)malloc(sizeof(t_neighbourList));
    neighbourList->next = NULL;
    neighbourList->particle = NULL;
    neighbourList->num_particles = 0;
    neighbourList->id = 0;

    t_neighbourList *prev_cell = neighbourList;

    for(int i = 1; i < matrix_size; i++){
        t_neighbourList *new_cell = (t_neighbourList*)malloc(sizeof(t_neighbourList));
        new_cell->next = NULL;
        new_cell->particle = NULL;
        new_cell->num_particles = 0;
        new_cell->id = i;

        prev_cell->next = new_cell;
        prev_cell = new_cell;
    }

    return neighbourList;
}

void clean_particle(t_neighbourList *neighbourList){
    t_neighbourList *current_cell = neighbourList;

    while(current_cell != NULL){
        t_neighbourList *next_cell = current_cell->next;
        t_point *current_particle = current_cell->particle;

        while (current_particle != NULL)
        {
            t_point *next_particle = current_particle->next;
            current_particle->next = NULL;
            current_particle = next_particle;
 
        }
        
        current_cell->particle = NULL;
        free(current_cell);
        current_cell = next_cell;
    }

}

void add_particle(t_neighbourList *neighbourList, t_point *particle){
    t_neighbourList *current_cell = neighbourList;
    int i = 0;
    

    float cell_x_index = floor(particle->cur_pos.x / CELL_SIZE);
    float cell_y_index = floor(particle->cur_pos.y / CELL_SIZE);
    float cell_z_index = floor(particle->cur_pos.z / CELL_SIZE);

    int cell_index = fmod(cell_x_index + cell_y_index * SIMULATION_SPACE + cell_z_index * SIMULATION_SPACE * SIMULATION_SPACE, SIMULATION_SPACE * SIMULATION_SPACE * SIMULATION_SPACE);

    while(i < cell_index){
        current_cell = current_cell->next;
        i++;
    }

    if(current_cell->particle == NULL){
        current_cell->particle = particle;
        current_cell->num_particles++;

        return;
    }

    t_point *current_particle = current_cell->particle;
    while(current_particle->next != NULL){
        current_particle = current_particle->next;
    }

    current_particle->next = particle;
    current_cell->num_particles++;

    return;
}

int detect_collision(t_neighbourList *neighbourList, t_point *particle){
    t_neighbourList *current_cell = neighbourList;
    int i = 0;
    int collision = 0;

    float cell_x_index = floor(particle->cur_pos.x / CELL_SIZE);
    float cell_y_index = floor(particle->cur_pos.y / CELL_SIZE);
    float cell_z_index = floor(particle->cur_pos.z / CELL_SIZE);

    // get the index of the cell where the particle should be
    int cell_index = fmod(cell_x_index + cell_y_index * SIMULATION_SPACE + cell_z_index * SIMULATION_SPACE * SIMULATION_SPACE, SIMULATION_SPACE * SIMULATION_SPACE * SIMULATION_SPACE);

    // get the cell
    while(i < cell_index){
        current_cell = current_cell->next;
        i++;
    }

    // If the particle is still inside the cell, continue simulation
    t_point *current_particle = current_cell->particle;
    while(current_particle != NULL){
        if(current_particle->id == particle->id)
            return collision;

        current_particle = current_particle->next;
    }


    // if the particle not found in the cell, recriate the list
    return 1;
}