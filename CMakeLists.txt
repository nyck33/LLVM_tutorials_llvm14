cmake_minimum_required(VERSION 3.12)
project(llvm14_tuts)

find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(toy toy.cpp)

target_link_libraries(toy PRIVATE LLVM)
