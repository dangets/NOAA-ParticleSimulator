#ifndef INLREADER_HPP
#define INLREADER_HPP

#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>


struct INLRecord {
    std::time_t time;
};


class INLReader {
    public:
        INLReader(const char *fname);
        virtual ~INLReader();

        INLRecord readRecord();
        INLRecord readRecord(std::size_t i);

    private:
        std::ifstream ins;

        std::size_t record_size;
        std::size_t num_records;
        std::size_t cur_record_i;

        //std::time_t t0;
        //std::time_t t1;
        //std::time_t tN;
        //double tDelta;

        std::size_t grid_num_x;
        std::size_t grid_num_y;
        std::size_t grid_num_z;
};


#endif /* end of include guard: INLREADER_HPP */
