#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstddef>



struct ParticleSource {
    int id;

    float pos_x;
    float pos_y;
    float pos_z;
};


struct Particle {
    float pos_x;
    float pos_y;
    float pos_z;

    // bool has_deposited;
    // source_id
    // age
    // pollutant_type, mass, etc...

    Particle(float x, float y, float z) :
        pos_x(x), pos_y(y), pos_z(z) { }

    std::string toString() {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << "x:" << pos_x << "  ";
        os << "y:" << pos_y << "  ";
        os << "z:" << pos_z;
        return os.str();
    }
};


struct Grid {
    const std::size_t num_x;
    const std::size_t num_y;
    const std::size_t num_z;

    Grid(std::size_t numx, std::size_t numy, std::size_t numz) :
        num_x(numx), num_y(numy), num_z(numz) { }
};

// wind_velocity[t][z][y][x] = (u, v, w)
// surface_height[y][x] = z
// particle_sources = vector<ParticleSource>();






void advect_particle(Particle &p) {
    p.pos_x += 1;
    p.pos_y += 1;
    p.pos_z += 0.5;
}


int main(int argc, char *argv[])
{
    // init settings/configuration
    // init meteorology grid

    Grid g = Grid(512, 512, 128);

    Particle p = Particle(1, 2, 3);

    for (int t=0; t<10; t++) {
        advect_particle(p);
        std::cout << p.toString() << std::endl;
    }

//    for (int timestep_i=0; timestep_i < num_timesteps; timestep_i++) {
//        // introduce new particles into the system
//
//        for (int pollutant_i=0; pollutant_i < num_pollutants; pollutant_i++) {
//            // skip terminated particles
//
//            // advect particle
//
//            // determine terrain height
//
//            // dry, wet, decay routines
//            // surface water transport
//        }
//    }

    return 0;
}

