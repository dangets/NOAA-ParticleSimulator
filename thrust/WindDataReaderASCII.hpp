#ifndef WINDDATAREADERASCII_HPP
#define WINDDATAREADERASCII_HPP

#include <iostream>

#include "WindDataReader.h"


class WindDataReaderASCII {
    public:
        WindDataReaderASCII(std::istream ins);

    private:
        float data;

        size_t num_x;
        size_t num_y;
        size_t num_z;
        size_t num_t;
};


WindDataReaderASCII::WindDataReaderASCII(std::istream ins)
{
    ins >> num_x;
    ins >> num_y;
    ins >> num_z;
    ins >> num_t;
}



#endif /* end of include guard: WINDDATAREADERASCII_HPP */

