#include <iostream>
#include <vector>
#include <fstream>
#include "Eigen/Dense"
#include "constants.h"

class Particle3D
{
private:
    Eigen::Vector3f position; // Position of the particle
    Eigen::Vector3f velocity; // Velocity of the particle
    float mass;               // Mass of the particle

public:
    // Constructors
    //Default constructor
    Particle3D() 
        : position(0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), mass(0.0f) 
    {
    
    }
    Particle3D(const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, float m) 
        : position(pos), velocity(vel), mass(m) 
    {
    
    }

    // Getters
    Eigen::Vector3f getPosition() const 
    { 
        return position; 
    }
    Eigen::Vector3f getVelocity() const 
    { 
        return velocity; 
    }
    float getMass() const 
    { 
        return mass; 
    }

    // Setters
    void setPosition(const Eigen::Vector3f& pos) 
    { 
        position = pos; 
    }
    void setVelocity(const Eigen::Vector3f& vel) 
    { 
        velocity = vel; 
    }
    void setMass(float m) 
    { 
        mass = m; 
    }

    //Function to calculate the LJ potential between two particles
    float calculateLJPotential(const Particle3D& particle,const float eps,const float sigma);

    //Function to calcualate the force update
    float forceUpdate(const Particle3D& particle, const float eps, const float sigma);
};

float Particle3D::calculateLJPotential(const Particle3D& particle,const float eps, const float sigma)
{
    // Potential V(x_i,x_j) = 4 * eps * ((sigma / xij) * 12 - (sigma / xij) * 6), xij = mod(x_i - x_j)
    Eigen::Vector3f r = particle.getPosition() - position;

    //Take the norm
    float xij = r.norm();

    float sigma_xij = sigma / xij;
    return (4 * eps * (std::pow(sigma_xij, 12) - (std::pow(sigma_xij, 6))));
    
}

float Particle3D::forceUpdate(const Particle3D& particle,const float eps, const float sigma)
{
    // F_2^LJ = 24 * eps * (sigma /  xij)^6 * 2[(sigma/xij)^6 - 1] 
    Eigen::Vector3f r = particle.getPosition() - position;

    //Take the norm
    float xij = r.norm();

    float sigma_xij = sigma / xij;
    return 24 * eps * std::pow((sigma_xij), 6) * 2 * (std::pow(sigma_xij,6) - 1);
}


int main(int argc, char* argv[])
{
    // Check if the correct number of command-line arguments is provided
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " time_steps step_size num_particles eps sigma" << std::endl;
        return 1; // Exit with an error code
    }

    // Parse command-line arguments
    int time_steps = std::atoi(argv[1]);
    float step_size = std::atof(argv[2]);
    int num_particles = std::atoi(argv[3]);
    float eps = std::atof(argv[4]);
    float sigma = std::atof(argv[5]);

    // Initialize particles and forces
    std::vector<Particle3D> particles(num_particles);
    std::vector<Eigen::Vector3f> forces(num_particles);

    // Perform simulation steps
    for (int step = 0; step < time_steps; ++step)
    {
        // Calculate forces between particles
        for (int i = 0; i < num_particles; ++i)
        {
            Particle3D& particle_i = particles[i];
            for (int j = i + 1; j < num_particles; ++j)
            {
                Particle3D& particle_j = particles[j];
                float force_ij = particle_i.forceUpdate(particle_j, eps, sigma);
                Eigen::Vector3f r = particle_j.getPosition() - particle_i.getPosition();
                Eigen::Vector3f f = force_ij * r.normalized(); // Calculate force vector
                forces[i] += f;
                forces[j] -= f; // Newton's third law
            }
        }

        // Update particle positions and velocities using Velocity Verlet algorithm
        for (int i = 0; i < num_particles; ++i)
        {
            Particle3D& particle = particles[i];

            // Calculate acceleration using the forces on the particle
            Eigen::Vector3f acceleration = forces[i] / particle.getMass();

            // Update velocity using half-step integration
            particle.setVelocity(particle.getVelocity() + 0.5f * step_size * acceleration);

            // Update position using full-step integration
            particle.setPosition(particle.getPosition() + step_size * particle.getVelocity());
        }

        // Clear forces for the next step (to remove accumulation of forces from previous time step) 
        std::fill(forces.begin(), forces.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));

        // Visualize particle positions (optional)
    }

    return 0;
}
