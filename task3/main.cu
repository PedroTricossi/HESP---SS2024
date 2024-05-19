#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

class Particle3D
{
private:
    Eigen::Vector3f position; // Position of the particle
    Eigen::Vector3f velocity; // Velocity of the particle
    float mass;               // Mass of the particle

public:
    // Constructors
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
    float calculateLJPotential(const Particle3D& particle, const float eps, const float sigma);

    //Function to calculate the force update
    float forceUpdate(const Particle3D& particle, const float eps, const float sigma);
};

float Particle3D::calculateLJPotential(const Particle3D& particle, const float eps, const float sigma)
{
    // Potential V(x_i,x_j) = 4 * eps * ((sigma / xij) * 12 - (sigma / xij) * 6), xij = mod(x_i - x_j)
    Eigen::Vector3f r = particle.getPosition() - position;

    //Take the norm
    float xij = r.norm();

    float sigma_xij = sigma / xij;
    return (4 * eps * (std::pow(sigma_xij, 12) - (std::pow(sigma_xij, 6))));

}

float Particle3D::forceUpdate(const Particle3D& particle, const float eps, const float sigma)
{
    // F_2^LJ = 24 * eps * (sigma /  xij)^6 * 2[(sigma/xij)^6 - 1] 
    Eigen::Vector3f r = particle.getPosition() - position;

    //Take the norm
    float xij = r.norm();

    float sigma_xij = sigma / xij;

    return 24 * eps * std::pow((sigma_xij), 6) * 2 * (std::pow(sigma_xij, 6) - 1) * (xij / (xij * xij));
}

void writeVTKFile(int step, int num_particles, std::vector<Particle3D>& particles) {
    std::ofstream simulationFile("simulation_" + std::to_string(step) + ".vtk");

    simulationFile << "# vtk DataFile Version 3.0 \n";
    simulationFile << "Lennard-Jones particle simulation \n";
    simulationFile << "ASCII \n";
    simulationFile << "DATASET UNSTRUCTURED_GRID \n";
    simulationFile << "POINTS " << num_particles << " float \n";

    for (int i = 0; i < num_particles; i++) {
        Eigen::Vector3f pos = particles[i].getPosition();
        simulationFile << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
    }

    simulationFile << "CELLS " << "0" << " " << "0" << "\n";
    simulationFile << "CELL_TYPES " << "0" << "\n";
    simulationFile << "POINT_DATA " << num_particles << "\n";
    simulationFile << "SCALARS mass float \n";
    simulationFile << "LOOKUP_TABLE default \n";

    for (int i = 0; i < num_particles; i++) {
        simulationFile << particles[i].getMass() << "\n";
    }

    simulationFile << "VECTORS velocity float \n";
    for (int i = 0; i < num_particles; i++) {
        Eigen::Vector3f vel = particles[i].getVelocity();
        simulationFile << vel[0] << " " << vel[1] << " " << vel[2] << "\n";
    }

}

__global__ void computeForces(Particle3D* particles, Eigen::Vector3f* forces, int num_particles, float eps, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        for (int j = 0; j < num_particles; ++j) {
            if (i != j) {
                float force_ij = particles[i].forceUpdate(particles[j], eps, sigma);
                Eigen::Vector3f r = particles[j].getPosition() - particles[i].getPosition();
                Eigen::Vector3f f = force_ij * r.normalized();
            /*
            We are using atomicAdd to avoid race condition from developing among threads. The same threads might try to
            compute the same variable thus leading to latency.
            */
                atomicAdd(&forces[i].x(), -f.x());
                atomicAdd(&forces[i].y(), -f.y());
                atomicAdd(&forces[i].z(), -f.z());
                atomicAdd(&forces[j].x(), f.x());
                atomicAdd(&forces[j].y(), f.y());
                atomicAdd(&forces[j].z(), f.z());
            }
        }
    }
}



__global__ void integrateParticles(Particle3D* particles, Eigen::Vector3f* forces, int num_particles, float step_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        Particle3D& particle = particles[i];
        Eigen::Vector3f acceleration = forces[i] / particle.getMass();
        particle.setPosition(particle.getPosition() + (step_size * particle.getVelocity()) + (step_size * step_size * acceleration * 0.5f));
        particle.setVelocity(particle.getVelocity() + (0.5f * step_size * acceleration));
    }
}

void simulateParticles(Particle3D* particles, Eigen::Vector3f* forces, int time_steps, int num_particles, float eps, float sigma, float step_size) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0

    int numberOfThreads = 256; // Declare and initialize numberOfThreads
    int numberOfBlocks = 32 * prop.multiProcessorCount; // Declare and initialize numberOfBlocks

    for (int step = 0; step < time_steps; ++step) {
        // Reset forces
        cudaMemset(forces, 0, num_particles * sizeof(Eigen::Vector3f));

        // Compute forces using CUDA
        computeForces << <numberOfBlocks, numberOfThreads >> > (particles, forces, num_particles, eps, sigma);
        cudaDeviceSynchronize();

        // Integrate particles using CUDA
        integrateParticles << <numberOfBlocks, numberOfThreads >> > (particles, forces, num_particles, step_size);
        cudaDeviceSynchronize();

        // Write the VTK file (done on the host)
        writeVTKFile(step + 1, num_particles, particles);
    }
}


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

   

    Particle3D* particles;
    Eigen::Vector3f* forces;

    cudaMallocManaged(&particles, num_particles * sizeof(Particle3D));
    cudaMallocManaged(&forces, num_particles * sizeof(Eigen::Vector3f));

    for (int i = 0; i < num_particles; ++i) {
        float x = (i == 0) ? i + 1 : i + 3;
        float y = x;
        float z = x;
        particles[i].setMass(1.0f);
        particles[i].setPosition(Eigen::Vector3f(x, y, z));
        particles[i].setVelocity(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
        forces[i] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    }

    writeVTKFile(0, num_particles, particles);

    simulateParticles(particles, forces, time_steps, num_particles, eps, sigma, step_size);

    cudaFree(particles);
    cudaFree(forces);

    return 0;
}
