#include <ctime>
#include <boost/date_time.hpp>

#include "ParticleSource.hpp"


int main(int argc, const char *argv[])
{
    using namespace boost::posix_time;
    using namespace boost::gregorian;

    ptime start(date(2012, Jun, 1), hours(12));
    ptime stop(date(2012, Jun, 1), hours(15));
    std::tm start_tm = to_tm(start);
    std::tm stop_tm = to_tm(stop);

    std::time_t start_time_t = std::mktime(&start_tm);
    std::time_t stop_time_t = std::mktime(&stop_tm);

    ParticleSource src(11.1, 22.2, 33.3, start_time_t, stop_time_t, 1);

    std::cout << "num lifetime particles: " << src.lifetimeParticlesReleased() << std::endl;

    return 0;
}
