#ifndef WINDDATA_HPP
#define WINDDATA_HPP

#include <vector>

using std::size_t;


struct WindData {
    WindData(size_t x, size_t y, size_t z, size_t t) :
        num_x(x), num_y(y), num_z(z), num_t(t),
        num_cells(x * y * z * t),
        u(num_cells), v(num_cells), w(num_cells)
    { }

    const size_t num_x;
    const size_t num_y;
    const size_t num_z;
    const size_t num_t;
    const size_t num_cells;

    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> w;

    // NOTE: does not check if x/y/z/t are within num_x/num_y/num_z/num_t
    size_t get_index(std::size_t x, std::size_t y, std::size_t z, std::size_t t) {
        return x + y * num_x + z * num_y * num_x + t * num_x * num_y * num_z;
    }
};


WindData WindDataFromASCII(const char * fname);
void WindDataPrint(const WindData &wd);


#endif /* end of include guard: WINDDATA_HPP */
