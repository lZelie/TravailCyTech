cmake_minimum_required(VERSION 3.28)
project(SurchargeOperateur)

set(CMAKE_CXX_STANDARD 26)

add_executable(SurchargeOperateur main.cpp
        fraction.cpp fraction.h
        garland.cpp garland.h
        light.cpp light.h
        time.cpp
        time.h)
