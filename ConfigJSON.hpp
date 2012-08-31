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

#ifndef CONFIGJSON_HPP
#define CONFIGJSON_HPP


#include <iostream>
#include <string>
#include <stdexcept>

#include <json-cpp/json.h>

#include "Config.hpp"
#include "ParticleSource.hpp"


Config ConfigFromJSON(std::istream &in) {
    Config cfg;

    Json::Value  root;   // will contain the root value after parsing.
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
        Position pos;
        Size size;


        std::string id = source_ids[i];
        Json::Value src = sources[id];
        pos.x = src["x"].asFloat();
        pos.y = src["y"].asFloat();
        pos.z = src["z"].asFloat();

        size.x = src.get("size_x", 0.1f).asFloat();
        size.y = src.get("size_y", 0.1f).asFloat();
        size.z = src.get("size_z", 0.1f).asFloat();

        unsigned int release_start = src["release_start"].asUInt();
        unsigned int release_stop  = src["release_stop"].asUInt();
        float        release_rate  = src["release_rate"].asFloat();

        // TODO: assign the source id
        cfg.particle_sources.push_back(
            ParticleSource(pos, size, release_start, release_stop, release_rate)
        );

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
