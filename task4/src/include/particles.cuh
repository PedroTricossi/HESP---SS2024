class Particle3D
{
private:
    float3 position; // Position of the particle
    float3 velocity; // Velocity of the particle
    float mass;      // Mass of the particle

public:
    // Constructors
    Particle3D()
            : mass(0.0f)
    {
        position.x = 0.0f;
        position.y = 0.0f;
        position.z = 0.0f;

        velocity.x = 0.0f;
        velocity.y = 0.0f;
        velocity.z = 0.0f;
    }

    Particle3D(float3 pos, float3 vel, float m)
            : position(pos), velocity(vel), mass(m)
    {
    }

    // Getters
    __host__ __device__ float3 getPosition() const { return position; }
    __host__ __device__ float3 getVelocity() const { return velocity; }
    __host__ __device__ float getMass() const { return mass; }

    // Setters
    __host__ __device__ void setPosition(float3 pos) { position = pos; }
    __host__ __device__ void setVelocity(float3 vel) { velocity = vel; }
    __host__ __device__ void setMass(float m) { mass = m; }

    // Function to calculate the force update
    __host__ __device__ float forceUpdate(const Particle3D& particle_j, const float eps, const float sigma){
        float3 r;

        r.x = particle_j.getPosition().x - position.x;
        r.y = particle_j.getPosition().y - position.y;
        r.z = particle_j.getPosition().z - position.z;

        float xij = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        float sigma_xij = sigma / xij;
        float sigma_xij_6 = powf(sigma_xij, 6);

        float f_scalar = 24 * eps * sigma_xij_6 * ( 2 * powf(sigma_xij, 6) - 1);

        return f_scalar;
    }
};

__global__ void compute_force_between_particles(Particle3D* particles, float3* forces, int num_particles, float eps, float sigma);

__global__ void apply_integrator_for_particle(Particle3D* particles, float3* forces, int num_particles, float step_size);