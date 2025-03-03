cmake_minimum_required(VERSION 3.31)
project(Raytracing1)

set(CMAKE_CXX_STANDARD 23)

#OpenGL
find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )

set( GLFW_BUILD_DOCS OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )
set( GLAD_GL "" )

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

add_executable(Raytracing1 main.cpp)
target_link_libraries(Raytracing1 ${OPENGL_gl_LIBRARY} glfw glm::glm-header-only)
