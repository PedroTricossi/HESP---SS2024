#!/bin/bash

# Define the initial number of particles
num_particles=1024

# Define the maximum number of particles
max_particles=1000000

# Define the base value for the power of 2
base=2

# Loop until the number of particles exceeds the maximum
while [ $num_particles -le $max_particles ]
do
    # Execute the particle simulation with the current number of particles
    ./particle_simulation 20 0.01 $num_particles 1 1 200 2

    # Multiply the number of particles by the base value
    num_particles=$((num_particles * base))
done