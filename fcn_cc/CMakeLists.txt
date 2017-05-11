cmake_minimum_required(VERSION 2.8)
project(external-project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(SOURCE_FILES loader.cpp)
set(EXECUTABLE external-project)

# Add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

find_package(TensorFlow REQUIRED)

find_package(Protobuf REQUIRED)

find_package(TIFF REQUIRED)
include_directories(${TIFF_INCLUDE_DIRS})
link_libraries(${TIFF_LIBRARIES})


find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

# set variables for external dependencies
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external" CACHE PATH "Location where external dependencies will installed")
set(DOWNLOAD_LOCATION "${EXTERNAL_DIR}/src" CACHE PATH "Location where external projects will be downloaded")
mark_as_advanced(EXTERNAL_DIR DOWNLOAD_LOCATION)



set(PROJECT_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIRS} ${EXTERNAL_DIR}/include)
set(PROJECT_LIBRARIES ${TENSORFLOW_LIBRARIES})

include_directories(${PROJECT_INCLUDE_DIRS})
add_executable(${EXECUTABLE} ${SOURCE_FILES})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})
#add_dependencies(${EXECUTABLE} ${PROJECT_DEPENDENCIES})