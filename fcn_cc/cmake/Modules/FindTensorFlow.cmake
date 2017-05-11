# Locates the tensorFlow library and include directories.

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TENSORFLOW_INCLUDE_DIR
        NAMES
        tensorflow
        third_party
        HINTS
        /usr/local/include/tensorflow
        /usr/include/tensorflow)

find_library(TENSORFLOW_LIBRARY NAMES tensorflow_cc
        HINTS
        /usr/lib
        /usr/local/lib)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY})
    set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)

#g++ -std=c++11 -o tLoader -I/usr/local/include/TENSORFLOW -I/usr/local/include/eigen3 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w  -L/usr/local/lib/libtensorflow_cc `pkg-config --cflags --libs protobuf`  -ltensorflow_cc loader.cpp
