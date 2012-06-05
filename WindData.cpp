#include <fstream>
#include <cstdio>

#include "WindData.hpp"


WindData WindDataFromASCII(const char * fname)
{
    std::ifstream ins;

    size_t num_x;
    size_t num_y;
    size_t num_z;
    size_t num_t;

    ins.open(fname);
    ins >> num_x;
    ins >> num_y;
    ins >> num_z;
    ins >> num_t;

    WindData wd(num_x, num_y, num_z, num_t);

    for (size_t t=0; t<num_t; t++) {
        size_t t_offset = t * num_z * num_y * num_x;
        for (size_t z=0; z<num_z; z++) {
            size_t z_offset = z * num_y * num_x;
            for (size_t y=0; y<num_y; y++) {
                size_t y_offset = y * num_x;
                for (size_t x=0; x<num_x; x++) {
                    size_t offset = x + y_offset + z_offset + t_offset;

                    ins >> wd.u[offset];
                    ins >> wd.v[offset];
                    ins >> wd.w[offset];
                }
            }
        }
    }

    ins.close();

    return wd;
}


void WindDataPrint(const WindData &wd)
{
    std::printf("%lu %lu %lu %lu\n", wd.num_x, wd.num_y, wd.num_z, wd.num_t);

    for (size_t t=0; t<wd.num_t; t++) {
        size_t t_offset = t * wd.num_z * wd.num_y * wd.num_x;
        for (size_t z=0; z<wd.num_z; z++) {
            size_t z_offset = z * wd.num_y * wd.num_x;
            for (size_t y=0; y<wd.num_y; y++) {
                size_t y_offset = y * wd.num_x;
                for (size_t x=0; x<wd.num_x; x++) {
                    size_t offset = x + y_offset + z_offset + t_offset;

                    std::printf("%7.2f ", wd.u[offset]);
                    std::printf("%7.2f ", wd.v[offset]);
                    std::printf("%7.2f ", wd.w[offset]);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
}

