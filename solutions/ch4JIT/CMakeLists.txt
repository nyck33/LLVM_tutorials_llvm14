cmake_minimum_required(VERSION 3.12)
project(Toy)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)

# Set the LLVM include directories and libraries
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Create an executable target
add_executable(toy toy.cpp)

# Link against LLVM libraries
llvm_map_components_to_libnames(llvm_libs core orcjit native)
target_link_libraries(toy ${llvm_libs})
