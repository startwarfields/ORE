cmake_minimum_required(VERSION 3.15) # Specify the minimum version of CMake required

project(MLTestBench) # Set the name of your project
LINK_DIRECTORIES(/usr/lib/onnxruntime)

add_executable(mltest main.cpp src/ml.h src/ml.cpp) # Create an executable named myexample from simple.cpp


target_link_libraries(mltest onnxruntime)

