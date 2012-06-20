#ifndef CONFIGJSON_HPP
#define CONFIGJSON_HPP


#include <iostream>
#include <string>
#include <json-cpp/json.h>

#include "Config.hpp"


Config ConfigFromJSON(std::istream &in) {
    Config cfg;

    Json::Value root;   // will contain the root value after parsing.
    Json::Reader reader;

    bool parsingSuccessful = reader.parse(in, root);
    if (!parsingSuccessful) {
        // report to the user the failure and their locations in the document.
        std::cerr  << "Failed to parse configuration: " << reader.getFormatedErrorMessages() << std::endl;
        throw std::runtime_error("failed to parse json configuration");
    }

    const Json::Value sources = root["particle_sources"];
    if (sources.size() == 0) {
        throw std::invalid_argument("no 'particle_sources' defined in config file");
    }

    unsigned int max_time = 0;
    Json::Value::Members source_ids = sources.getMemberNames();
    for (size_t i=0; i<source_ids.size(); ++i) {
        // TODO: validate that the required fields exist

        std::string id = source_ids[i];
        Json::Value src = sources[id];
        float x = src["x"].asFloat();
        float y = src["y"].asFloat();
        float z = src["z"].asFloat();
        time_t release_start = src["release_start"].asUInt();
        time_t release_stop = src["release_stop"].asUInt();
        float  release_rate = src["release_rate"].asFloat();

        float dx = src.get("dx", 0.1f).asFloat();
        float dy = src.get("dy", 0.1f).asFloat();
        float dz = src.get("dz", 0.1f).asFloat();

        // TODO: assign source id
        cfg.particle_sources.push_back(ParticleSource(x, y, z, release_start, release_stop, release_rate, dx, dy, dz));

        if (release_stop > max_time) {
            max_time = release_stop;
        }
    }

    cfg.num_seconds = root.get("num_seconds", max_time).asUInt();

    return cfg;
}


std::string ConfigToJSON(const Config &cfg) {
    return "";
}


#endif /* end of include guard: CONFIGJSON_HPP */
