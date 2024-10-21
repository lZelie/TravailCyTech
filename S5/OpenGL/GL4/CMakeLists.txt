cmake_minimum_required(VERSION 3.30)
project(GL4)

set(CMAKE_CXX_STANDARD 26)

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
if (NOT OPENGL_FOUND)
    message(ERROR "OPENGL not found!")
endif (NOT OPENGL_FOUND)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
if (NOT GLEW_FOUND)
    message(ERROR "GLEW not found!")
endif (NOT GLEW_FOUND)

include_directories(libraries/include)

add_executable(GL4 main.cpp
        app.cpp
        camera.cpp
        shader_class.cpp
        vao.cpp
        vbo.cpp
        texture.cpp
        texture.h
        fbo.cpp
        fbo.h
)

target_link_libraries(GL4 ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
