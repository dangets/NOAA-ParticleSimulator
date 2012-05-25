#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <ctime>


typedef float LatLon;
typedef float real;
typedef unsigned int uint;


/**
 * interpret a possibly 2 digit year into a 4 digit year
 * the threshold value is 1940-2040
 */
static int get4DigitYear(int year) {
    int year4;
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


// logic copied from library/hysplit/metset.f

void readINLFile(std::istream &ins)
{
    static const unsigned int MAX_HEADER_LENGTH = 8000;
    std::ostream &log = std::cout;

    std::tm tmp_tm = { 0 };
    tmp_tm.tm_isdst = -1;
    tmp_tm.tm_sec = 0;

    std::time_t t0;
    std::time_t t1;
    std::time_t tN;
    double tDelta;

    char buff[256] = { 0 };

    // 'label' variables
    uint year, month, day, hour, forecast_hour;
    uint num_x, num_y;
    std::stringstream label;
    std::stringstream header;
    std::string var;

    ins.get(buff, 51); label.str(buff);
    ins.get(buff, 109); header.str(buff);

    //log << "label:" << label.str() << std::endl;
    //log << "header:" << header.str() << std::endl;

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
        return;
    }
    // DREC(KG,KT)%TYPE=2

    // record header variables
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

    t0 = mktime(&tmp_tm);
    printTime(tmp_tm);

    // this may not be necessary...
    assert(header_length <= MAX_HEADER_LENGTH);

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

    log << "records_per_time_period:" << records_per_time_period << std::endl;
    //log << "current file position:" << ins.tellg() << std::endl;

    // skip to record t1
    std::size_t offset = record_length * records_per_time_period;
    ins.seekg(offset, std::ios::beg);
    log << "current file position:" << ins.tellg() << std::endl;
    ins.get(buff, 51);  label.str(buff);  label.clear();
    ins.get(buff, 109); header.str(buff); header.clear();

    //log << "offset:" << offset << std::endl;
    //log << "label:" << label.str() << std::endl;
    //log << "header:" << header.str() << std::endl;

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

    header.get(buff, 5); model_id = buff;
    header.get(buff, 4); start_hour = std::atoi(buff);
    header.get(buff, 3); start_minute = std::atoi(buff);

    //log << "----- Label ------------" << std::endl;
    //log << "year: " << year << std::endl;
    //log << "month: " << month << std::endl;
    //log << "day: " << day << std::endl;
    //log << "hour: " << hour << std::endl;
    //log << "forecast_hour: " << forecast_hour << std::endl;
    //log << "num_x: " << num_x << std::endl;
    //log << "num_y: " << num_y << std::endl;
    //log << "var: " << var << std::endl;

    //log << "----- Header -----------" << std::endl;
    //log << "model_id: " << model_id << std::endl;
    //log << "start_hour: " << start_hour << std::endl;
    //log << "start_minute: " << start_minute << std::endl;

    tmp_tm.tm_year = get4DigitYear(year) - 1900;
    tmp_tm.tm_mon = month - 1;
    tmp_tm.tm_mday = day;
    tmp_tm.tm_hour = hour;
    tmp_tm.tm_min = start_minute;
    tmp_tm.tm_sec = 0;

    t1 = mktime(&tmp_tm);
    printTime(tmp_tm);

    // find time delta between t0 and t1
    tDelta = std::difftime(t1, t0);
    //log << "tDelta:" << tDelta << std::endl;
    log << "current file position:" << ins.tellg() << std::endl;

    // loop over all time periods using the offset jump size
    // verify that all minute deltas are the same
    int t=2;
    std::time_t tLast = t1;
    while (true) {
        offset = t * record_length * records_per_time_period;
        ins.seekg(offset, std::ios::beg);
        //log << "current file position:" << ins.tellg() << std::endl;
        ins.get(buff, 51); label.str(buff); label.clear();
        ins.get(buff, 109); header.str(buff); header.clear();
        //log << "current file position:" << ins.tellg() << std::endl;
        if (ins.eof()) {
            break;
        }

//        label.get(buff, 3); year  = std::atoi(buff);
//        label.get(buff, 3); month = std::atoi(buff);
//        label.get(buff, 3); day   = std::atoi(buff);
//        label.get(buff, 3); hour  = std::atoi(buff);
//
//        header.get(buff, 5); model_id = buff;
//        header.get(buff, 4); start_hour = std::atoi(buff);
//        header.get(buff, 3); start_minute = std::atoi(buff);
//
//        //log << "----- XXXXXX -----------" << std::endl;
//        //log << "offset:" << offset << std::endl;
//        //log << "label:" << label.str() << std::endl;
//        //log << "header:" << header.str() << std::endl;
//        //log << "----- Header -----------" << std::endl;
//        //log << "model_id: " << model_id << std::endl;
//        //log << "start_hour: " << start_hour << std::endl;
//        //log << "start_minute: " << start_minute << std::endl;
//
//        tmp_tm.tm_year = get4DigitYear(year) - 1900;
//        tmp_tm.tm_mon = month - 1;
//        tmp_tm.tm_mday = day;
//        tmp_tm.tm_hour = hour;
//        tmp_tm.tm_min = start_minute;
//        tmp_tm.tm_sec = 0;
//
//        tN = mktime(&tmp_tm);
//        printTime(tmp_tm);
//        if (std::difftime(tN, tLast) != tDelta) {
//            log << "tDelta inconsistent:" << std::difftime(tN, tLast) << std::endl;
//        }
//
        tLast = tN;
        t++;
    }

//    log << "t:" << t << std::endl;
}


void extractRecord(std::istream &ins)
{
    static const unsigned int MAX_HEADER_LENGTH = 8000;
    std::ostream &log = std::cout;

    std::tm tmp_tm = { 0 };
    tmp_tm.tm_isdst = -1;
    tmp_tm.tm_sec = 0;

    std::time_t t0;
    double tDelta;

    char buff[256] = { 0 };
    std::stringstream label;
    std::stringstream header;

    // 'label' variables
    uint year, month, day, hour, forecast_hour;
    uint level;
    uint num_x, num_y;
    std::string var;
    //READ(LABEL,'(6I2,2X,A4,I4,2E14.7)') IY,IM,ID,IH,IC,LL,VARB,NEXP,PREC,VAR1

    ins.get(buff, 51); label.str(buff);
    ins.get(buff, 109); header.str(buff);

    //log << "label:" << label.str() << std::endl;
    //log << "header:" << header.str() << std::endl;

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

    // record header variables
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

    t0 = mktime(&tmp_tm);
    printTime(tmp_tm);

    // this may not be necessary...
    assert(header_length <= MAX_HEADER_LENGTH);

    uint record_length = (grid_num_x * grid_num_y) + 50;
    uint num_index_records = (header_length / (grid_num_x * grid_num_y)) + 1;
    uint records_per_time_period = num_index_records;

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

    log << "records_per_time_period:" << records_per_time_period << std::endl;

    //READ(KUNIT,REC=JREC,ERR=920)LABEL,CDATA
    //READ(LABEL,'(6I2,2X,A4,I4,2E14.7)') IY,IM,ID,IH,IC,LL,VARB,NEXP,PREC,VAR1
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <INLfile>\n", argv[0]);
        std::exit(1);
    }

    std::ifstream infile;

    infile.open(argv[1]);
    //readINLFile(infile);
    extractRecord(infile);
    infile.close();

    return 0;
}
