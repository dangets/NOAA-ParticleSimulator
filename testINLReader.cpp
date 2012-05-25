
#include <cstdio>
#include <cstdlib>

#include "INLReader.hpp"


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <INLfile>\n", argv[0]);
        std::exit(1);
    }

    INLReader rdr(argv[1]);

    INLRecord rec = rdr.readRecord();

    return 0;
}
