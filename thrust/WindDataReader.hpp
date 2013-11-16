#ifndef WINDDATAREADER_HPP
#define WINDDATAREADER_HPP


class WindDataReader {
    public:
        virtual ~WindDataReader() { }
        virtual float get(float x, float y, float z, float t) = 0;
};



#endif /* end of include guard: WINDDATAREADER_HPP */

