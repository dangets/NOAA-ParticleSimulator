#include <ctime>
#include <vector>

namespace hysplit {
    typedef float real;
    typedef unsigned int uint;

    const uint MAX_LEVELS = 75;
    const uint MAX_VARIABLES = 35;
    const uint MAX_GRIDS = 12;
    const uint MAX_PERIODS = 12;

    struct SourceLocation {
        real latitude;
        real longitude;
        real elevation;
        real emission_rate;   // (units/hr)
        real emission_area;   // (m^2)
        std::time_t start_datetime;
        real curpos_x;
        real curpos_y;
        real curpos_z;
        real prev_height;   // previous height for line sources
        uint meteo_grid_i;
    };

    struct Grid {
        char model_id[4];       // model data identification
        uint id;                // grid identification
        real pole_latitude;
        real pole_longitude;
        real ref_latitude;
        real ref_longitude;
        real size;              // "grid" size at ref lat lon
        real orientation;       // TODO
        real tangent_latitude;
        real sync_x;
        real sync_y;
        real sync_lat;
        real sync_lon;

        uint num_west_east;
        uint num_south_north;
        uint num_levels;

        bool is_latlon;         // flag if input grid is lat-lon
        bool crosses_prime;     // lat/lon grid crosses the prime meridian
        bool is_global;         // flag if latlon subgrid is global
        bool is_input_global;   // flag if latlon input grid is global

        uint subgrid_lx1;
        uint subgrid_ly1;
        uint subgrid_lx_range;
        uint subgrid_ly_range;
        real subgrid_lx_center;
        real subgrid_ly_center;
    };

    struct GenericModel {
        // probably should be all const
        uint num_simulation_hours;
        uint vertical_velocity_remapping;
        real max_input_height;
        uint num_grids;
        uint num_data_periods;

        // default constructor
        GenericModel() :
            num_simulation_hours(48), vertical_velocity_remapping(0),
            max_input_height(10000.0), num_grids(1), num_data_periods(1)
        { }
    };


    struct GridFile {
        std::string dir;
        std::string fname;
        uint record_length;
        uint num_records;
        std::time_t start_time;
        std::time_t end_time;
    };


    struct DataRecord {
        enum {
            SIGMA,
            PRESSURE,
            TERRAIN,
        } z_flag;

        real z_m_delta;     // scaling variable for input data
        real height[MAX_LEVELS];
        uint num_variables[MAX_LEVELS];
        char variable_id[MAX_VARIABLES][MAX_LEVELS][4];
        int  checksum[MAX_VARIABLES][MAX_LEVELS];
        uint records_per_time_period;
        uint period_delta_minutes;
        uint num_records_offset;
        uint accumulation_cycle_minutes;
        uint vertical_extrapolation_i;
        uint deformation_horizontal_mixing;
        uint vertical_profile_averaging;
        uint pbl_stability_derived_from;
        uint pbl_turbulence_method;

        enum {
            NO,
            ARW,
            NMM,
            STILT,
        } wrf;

        real trophospheric_vertical_mix_scale_factor;
        bool is_velocity_temporally_averaged;
        bool is_vertical_motion;
        bool is_surface_wind;
        bool is_surface_temp;
        bool is_mixed_layer_depth_avaiable;
        bool is_downward_shortwave_flux;
        bool is_exchange_coefficient;
        bool is_sensible_heat_flux;
        bool is_momentum_flux;
        bool is_friction_velocity;
        bool is_friction_temperature;
        bool is_velocity_variance;
        bool is_tke_field;
        bool is_surface_terrain_height;
        bool is_surface_pressure;
        bool is_upper_level_pressure;
        bool is_humidity_expressed_as_specific;
        bool is_upper_level_humidity[MAX_LEVELS];
        bool is_upper_level_wvelocity[MAX_LEVELS];
    };
}

namespace hs = hysplit;


static const int MAX_PARTICLES = 1000;
static const int MAX_POLLUNTANTS_PER_PARTICLE = 90;



int main(int argc, char *argv[])
{
    hs::real particle_xpos[MAX_PARTICLES];
    hs::real particle_ypos[MAX_PARTICLES];
    hs::real particle_zpos[MAX_PARTICLES];

    hs::real particle_sigma_h[MAX_PARTICLES];
    hs::real particle_sigma_u[MAX_PARTICLES];
    hs::real particle_sigma_v[MAX_PARTICLES];
    hs::real particle_sigma_w[MAX_PARTICLES];

    hs::real particle_source_mass[MAX_POLLUNTANTS_PER_PARTICLE][MAX_PARTICLES];
    hs::real mass_summation[MAX_POLLUNTANTS_PER_PARTICLE];

    hs::uint particle_age[MAX_PARTICLES];
    hs::uint particle_distribution_hdwp[MAX_PARTICLES];
    hs::uint particle_type[MAX_PARTICLES];

    hs::uint particle_meteo_grid[MAX_PARTICLES];
    hs::uint particle_sort_arr[MAX_PARTICLES];

    std::vector<hs::SourceLocation> source_locations;

    // TODO: initialize source_locations

    hs::uint num_starting_locations;
    hs::uint simulation_duration_hrs;
    hs::uint num_grids;
    hs::uint num_data_periods;
    hs::real max_scaling_height;
    hs::real max_input_height;
    bool do_back_integration;
    hs::uint vertical_velocity_remapping;


    return 0;
}

