cmake_minimum_required(VERSION 3.30)
project(Application)

set(CMAKE_CXX_STANDARD 26)

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
if (NOT GLUT_FOUND)
    message(ERROR "Glut not found")
endif ()

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR "OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
if(NOT GLEW_FOUND)
    message(ERROR "GLEW not found!")
endif (NOT GLEW_FOUND)

include_directories(.)

add_executable(Application
        Application/Application.cpp
        Application/GxParticle.cpp
        Application/GxUpdater.cpp
        Application/GxWorld.cpp
        Quantum/QmParticle.cpp
        Quantum/QmBody.cpp
        Quantum/QmWorld.cpp
        Quantum/QmForceRegistry.cpp
        Quantum/QmForceRegistry.h
        Quantum/QmForceGenerator.h
        Quantum/QmDrag.cpp
        Quantum/QmDrag.h
        Quantum/QmGravity.cpp
        Quantum/QmGravity.h
        Quantum/QmMagnetism.cpp
        Quantum/QmMagnetism.h
        Quantum/QmFixedMagnetism.cpp
        Quantum/QmFixedMagnetism.h
        Quantum/QmSpring.cpp
        Quantum/QmSpring.h
        Quantum/QmFixedSpring.cpp
        Quantum/QmFixedSpring.h
        Quantum/QmContact.cpp
        Quantum/QmContact.h
        Quantum/QmAABB.cpp
        Quantum/QmAABB.h
        Quantum/QmHalfspace.cpp
        Quantum/QmHalfspace.h
        Application/GxPlane.cpp
        Application/GxPlane.h
)

target_link_libraries(Application ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})