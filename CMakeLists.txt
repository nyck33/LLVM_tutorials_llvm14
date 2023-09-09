cmake_minimum_required(VERSION 3.15) 

project(LlvmToyCompiler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${LLVM_INCLUDE_DIRS})

add_executable(toy toy.cpp)

