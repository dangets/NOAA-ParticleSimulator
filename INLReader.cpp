#include <iostream>
#include <sstream>

#include "INLReader.hpp"


typedef float LatLon;
typedef float real;
typedef unsigned int uint;


/**
 * interpret a possibly 2 digit year into a 4 digit year
 * the threshold value is 1940-2040
 */
static int get4DigitYear(int year) {
    uint year4;
    if (year <= 100) {
        year4 = (year + 99) % 100 + 1901;
        if (year4 < 1940) {
            // assume no data prior to 1940 - shift year by a century
            year4 += 100;
        }
    } else {
        year4 = year;
    }

    return year4;
}


static void printTime(std::tm &time) {
    char buff[80];
    strftime(buff, 80, "%Y-%m-%d %H:%M:%S", &time);
    std::cout << buff << std::endl;
}





INLReader::INLReader(const char *fname)
    : record_size(0), num_records(0), cur_record_i(0)
{
    ins.open(fname);
}


INLReader::~INLReader()
{
    ins.close();
}


INLRecord INLReader::readRecord()
{
    return readRecord(cur_record_i);
}


INLRecord INLReader::readRecord(std::size_t i)
{
    std::ostream &log = std::cout;

    INLRecord rec;

    char buff[256] = { 0 };
    std::stringstream label;
    std::stringstream header;

    std::tm tmp_tm = { 0 };
    tmp_tm.tm_isdst = -1;
    tmp_tm.tm_sec = 0;

    ins.get(buff, 51); label.str(buff);
    ins.get(buff, 109); header.str(buff);

    //log << "label:" << label.str() << std::endl;
    //log << "header:" << header.str() << std::endl;

    // 'label' variables
    uint year, month, day, hour, forecast_hour;
    uint num_x, num_y;
    std::string var;

    // parse 'label'
    label.get(buff, 3); year  = std::atoi(buff);
    label.get(buff, 3); month = std::atoi(buff);
    label.get(buff, 3); day   = std::atoi(buff);
    label.get(buff, 3); hour  = std::atoi(buff);
    label.get(buff, 3); forecast_hour = std::atoi(buff);
    label.get(buff, 3); // 2 unused chars
    label.get(buff, 2); num_x = std::atoi(buff);
    label.get(buff, 2); num_y = std::atoi(buff);
    label.get(buff, 5); var = buff;

    log << "----- Label ------------" << std::endl;
    log << "year: " << year << std::endl;
    log << "month: " << month << std::endl;
    log << "day: " << day << std::endl;
    log << "hour: " << hour << std::endl;
    log << "forecast_hour: " << forecast_hour << std::endl;
    log << "num_x: " << num_x << std::endl;
    log << "num_y: " << num_y << std::endl;
    log << "var: " << var << std::endl;

    if (var != "INDX") {
        // TODO: implement reading "old" format
        //  this sets DREC(KG,KT)%TYPE=1
        throw;
    }
    // DREC(KG,KT)%TYPE=2

    // 'header' variables
    std::string model_id;
    uint start_hour;
    uint start_minute;
    LatLon pole_lat, pole_lon;
    LatLon ref_lat, ref_lon;
    LatLon tangent_lat;
    real grid_size, grid_orientation;
    real sync_x, sync_y;
    LatLon sync_lat, sync_lon;
    uint grid_num_x, grid_num_y, num_lvls;
    uint zflag;     // 1:sigma  2:pressure  3:terrain
    uint header_length;

    // parse 'header'
    header.get(buff, 5); model_id = buff;
    header.get(buff, 4); start_hour = std::atoi(buff);
    header.get(buff, 3); start_minute = std::atoi(buff);
    header.get(buff, 8); pole_lat = std::atof(buff);
    header.get(buff, 8); pole_lon = std::atof(buff);
    header.get(buff, 8); ref_lat = std::atof(buff);
    header.get(buff, 8); ref_lon = std::atof(buff);
    header.get(buff, 8); grid_size = std::atof(buff);
    header.get(buff, 8); grid_orientation = std::atof(buff);
    header.get(buff, 8); tangent_lat = std::atof(buff);
    header.get(buff, 8); sync_x = std::atof(buff);
    header.get(buff, 8); sync_y = std::atof(buff);
    header.get(buff, 8); sync_lat = std::atof(buff);
    header.get(buff, 8); sync_lon = std::atof(buff);
    header.get(buff, 8); // unused / reserved
    header.get(buff, 4); grid_num_x = std::atoi(buff);
    header.get(buff, 4); grid_num_y = std::atoi(buff);
    header.get(buff, 4); num_lvls = std::atoi(buff);
    header.get(buff, 3); zflag = std::atoi(buff);
    header.get(buff, 5); header_length = std::atoi(buff);

    log << "----- Header -----------" << std::endl;
    log << "model_id: " << model_id << std::endl;
    log << "start_hour: " << start_hour << std::endl;
    log << "start_minute: " << start_minute << std::endl;
    log << "pole_lat: " << pole_lat << std::endl;
    log << "pole_lon: " << pole_lon << std::endl;
    log << "ref_lat: " << ref_lat << std::endl;
    log << "ref_lon: " << ref_lon << std::endl;
    log << "grid_size: " << grid_size << std::endl;
    log << "grid_orientation: " << grid_orientation << std::endl;
    log << "tangent_lat: " << tangent_lat << std::endl;
    log << "sync_x: " << sync_x << std::endl;
    log << "sync_y: " << sync_y << std::endl;
    log << "sync_lat: " << sync_lat << std::endl;
    log << "sync_lon: " << sync_lon << std::endl;
    log << "grid_num_x: " << grid_num_x << std::endl;
    log << "grid_num_y: " << grid_num_y << std::endl;
    log << "num_lvls: " << num_lvls << std::endl;
    log << "zflag: " << zflag << std::endl;
    log << "header_length: " << header_length << std::endl;

    tmp_tm.tm_year = get4DigitYear(year) - 1900;
    tmp_tm.tm_mon = month - 1;
    tmp_tm.tm_mday = day;
    tmp_tm.tm_hour = hour;
    tmp_tm.tm_min = start_minute;

    rec.time = mktime(&tmp_tm);
    printTime(tmp_tm);

    uint record_length = (grid_num_x * grid_num_y) + 50;
    uint num_index_records = (header_length / (grid_num_x * grid_num_y)) + 1;
    uint records_per_time_period = num_index_records;

    //log << "current file position:" << ins.tellg() << std::endl;

    //log << "record_length: " << record_length << std::endl;
    //log << "num_index_records: " << num_index_records << std::endl;
    for (uint lvl_i = 0; lvl_i < num_lvls; lvl_i++) {
        real lvl_height;
        uint num_vars;

        ins.get(buff, 7); lvl_height = std::atof(buff);
        ins.get(buff, 3); num_vars = std::atoi(buff);
        log << "---- lvl_i:" << lvl_i << "  lvl_height:" << lvl_height << "  num_vars:" << num_vars << std::endl;
        for (uint var_i=0; var_i<num_vars; var_i++) {
            std::string var;
            uint checksum;
            ins.get(buff, 5); var = buff;
            ins.get(buff, 4); checksum = std::atoi(buff);
            ins.ignore(1);
            log << "var_i:" << var_i << "  var:" << var << "  checksum:" << checksum << std::endl;

            records_per_time_period++;
        }
    }

    record_size = record_length * records_per_time_period;

    log << "records_per_time_period:" << records_per_time_period << std::endl;
    log << "record_size:" << record_size << std::endl;

    return rec;
}
