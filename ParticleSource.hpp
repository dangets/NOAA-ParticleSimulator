/*
   Author: Danny George
   High Performance Simulation Laboratory
   Boise State University
 
   Permission is hereby granted, free of charge, to any person obtaining a copy of
   this software and associated documentation files (the "Software"), to deal in
   the Software without restriction, including without limitation the rights to
   use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is furnished to do
   so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

#ifndef PARTICLESOURCE_HPP
#define PARTICLESOURCE_HPP


struct Position {
    float x;
    float y;
    float z;
};

struct Size {
    float x;
    float y;
    float z;
};


struct ParticleSource {
    ParticleSource(const Position &pos, const Size &siz,
            const unsigned int &start, const unsigned int &stop, float rate)
        : position(pos), size(siz),
          release_start(start), release_stop(stop), release_rate(rate)
    { }

    Position position;
    Size     size;

    unsigned int release_start;    // relative seconds
    unsigned int release_stop;
    float        release_rate;     // particles per second
};



#endif /* end of include guard: PARTICLESOURCE_HPP */
