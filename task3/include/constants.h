#pragma once

#define SIMULATION_SPACE 10 // 10 x 10 x 10
#define ARGON_VAN_DER_WAALS_RADIOUS 1.88  // Angstrom
#define CELL_SIZE (ARGON_VAN_DER_WAALS_RADIOUS * 2.5) // Angstrom
#define NUM_PARTICLES 10
#define DT 0.042 // approx 1/24 seconds
#define NUM_INTERACTION 10 // approx 10 seconds