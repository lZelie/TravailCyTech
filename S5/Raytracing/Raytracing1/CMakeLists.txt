cmake_minimum_required(VERSION 3.31)
project(Raytracing1)

set(CMAKE_CXX_STANDARD 23)

#OpenGL
find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )

#GLFW
find_package(glfw QUIET)
if (NOT glfw_found)
    include(FetchContent)
    FetchContent_Declare(GLFW GIT_REPOSITORY https://github.com/glfw/glfw.git)
    FetchContent_MakeAvailable(GLFW)
endif ()

# GLM
find_package(glm CONFIG QUIET)
if (NOT glm_FOUND)
    include(FetchContent)
    FetchContent_Declare(GLM GIT_REPOSITORY https://github.com/icaven/glm.git)
    FetchContent_MakeAvailable(GLM)
endif ()

# GLEW
find_package(GLEW QUIET)
if (NOT GLEW_FOUND)
    include(FetchContent)
    SET(BUILD_UTILS OFF)
    FetchContent_Declare(
            GLEW
            URL            https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.zip
            SOURCE_DIR     ${CMAKE_SOURCE_DIR}/external/glew
    )
    FetchContent_GetProperties(glew)
    if(NOT glew_POPULATED)
        FetchContent_MakeAvailable(glew)
        add_subdirectory(${CMAKE_SOURCE_DIR}/external/glew/build/cmake)
    endif()
endif (NOT GLEW_FOUND)

add_executable(Raytracing1 main.cpp
        shader_class.cpp
        shader_class.h
        vao.h
        vao.cpp
        vbo.cpp
        vbo.h
        camera.cpp
        camera.h)
target_include_directories(Raytracing1 PRIVATE "${CMAKE_SOURCE_DIR}/include" external/glew/include)
target_link_libraries(Raytracing1 ${OPENGL_gl_LIBRARY} glfw glm::glm-header-only glew_s)
