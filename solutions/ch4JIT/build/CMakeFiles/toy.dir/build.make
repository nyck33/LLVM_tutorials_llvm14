# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build

# Include any dependencies generated for this target.
include CMakeFiles/toy.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/toy.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/toy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/toy.dir/flags.make

CMakeFiles/toy.dir/toy.cpp.o: CMakeFiles/toy.dir/flags.make
CMakeFiles/toy.dir/toy.cpp.o: ../toy.cpp
CMakeFiles/toy.dir/toy.cpp.o: CMakeFiles/toy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/toy.dir/toy.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/toy.dir/toy.cpp.o -MF CMakeFiles/toy.dir/toy.cpp.o.d -o CMakeFiles/toy.dir/toy.cpp.o -c /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/toy.cpp

CMakeFiles/toy.dir/toy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/toy.dir/toy.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/toy.cpp > CMakeFiles/toy.dir/toy.cpp.i

CMakeFiles/toy.dir/toy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/toy.dir/toy.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/toy.cpp -o CMakeFiles/toy.dir/toy.cpp.s

# Object files for target toy
toy_OBJECTS = \
"CMakeFiles/toy.dir/toy.cpp.o"

# External object files for target toy
toy_EXTERNAL_OBJECTS =

toy: CMakeFiles/toy.dir/toy.cpp.o
toy: CMakeFiles/toy.dir/build.make
toy: /usr/lib/llvm-14/lib/libLLVMCore.a
toy: /usr/lib/llvm-14/lib/libLLVMOrcJIT.a
toy: /usr/lib/llvm-14/lib/libLLVMX86CodeGen.a
toy: /usr/lib/llvm-14/lib/libLLVMX86AsmParser.a
toy: /usr/lib/llvm-14/lib/libLLVMX86Desc.a
toy: /usr/lib/llvm-14/lib/libLLVMX86Disassembler.a
toy: /usr/lib/llvm-14/lib/libLLVMX86Info.a
toy: /usr/lib/llvm-14/lib/libLLVMPasses.a
toy: /usr/lib/llvm-14/lib/libLLVMCoroutines.a
toy: /usr/lib/llvm-14/lib/libLLVMipo.a
toy: /usr/lib/llvm-14/lib/libLLVMFrontendOpenMP.a
toy: /usr/lib/llvm-14/lib/libLLVMIRReader.a
toy: /usr/lib/llvm-14/lib/libLLVMAsmParser.a
toy: /usr/lib/llvm-14/lib/libLLVMLinker.a
toy: /usr/lib/llvm-14/lib/libLLVMObjCARCOpts.a
toy: /usr/lib/llvm-14/lib/libLLVMVectorize.a
toy: /usr/lib/llvm-14/lib/libLLVMExecutionEngine.a
toy: /usr/lib/llvm-14/lib/libLLVMJITLink.a
toy: /usr/lib/llvm-14/lib/libLLVMOrcTargetProcess.a
toy: /usr/lib/llvm-14/lib/libLLVMOrcShared.a
toy: /usr/lib/llvm-14/lib/libLLVMRuntimeDyld.a
toy: /usr/lib/llvm-14/lib/libLLVMAsmPrinter.a
toy: /usr/lib/llvm-14/lib/libLLVMDebugInfoMSF.a
toy: /usr/lib/llvm-14/lib/libLLVMInstrumentation.a
toy: /usr/lib/llvm-14/lib/libLLVMGlobalISel.a
toy: /usr/lib/llvm-14/lib/libLLVMSelectionDAG.a
toy: /usr/lib/llvm-14/lib/libLLVMCodeGen.a
toy: /usr/lib/llvm-14/lib/libLLVMBitWriter.a
toy: /usr/lib/llvm-14/lib/libLLVMTarget.a
toy: /usr/lib/llvm-14/lib/libLLVMScalarOpts.a
toy: /usr/lib/llvm-14/lib/libLLVMAggressiveInstCombine.a
toy: /usr/lib/llvm-14/lib/libLLVMInstCombine.a
toy: /usr/lib/llvm-14/lib/libLLVMTransformUtils.a
toy: /usr/lib/llvm-14/lib/libLLVMAnalysis.a
toy: /usr/lib/llvm-14/lib/libLLVMProfileData.a
toy: /usr/lib/llvm-14/lib/libLLVMDebugInfoDWARF.a
toy: /usr/lib/llvm-14/lib/libLLVMObject.a
toy: /usr/lib/llvm-14/lib/libLLVMBitReader.a
toy: /usr/lib/llvm-14/lib/libLLVMTextAPI.a
toy: /usr/lib/llvm-14/lib/libLLVMCFGuard.a
toy: /usr/lib/llvm-14/lib/libLLVMCore.a
toy: /usr/lib/llvm-14/lib/libLLVMRemarks.a
toy: /usr/lib/llvm-14/lib/libLLVMBitstreamReader.a
toy: /usr/lib/llvm-14/lib/libLLVMMCParser.a
toy: /usr/lib/llvm-14/lib/libLLVMMCDisassembler.a
toy: /usr/lib/llvm-14/lib/libLLVMMC.a
toy: /usr/lib/llvm-14/lib/libLLVMBinaryFormat.a
toy: /usr/lib/llvm-14/lib/libLLVMDebugInfoCodeView.a
toy: /usr/lib/llvm-14/lib/libLLVMSupport.a
toy: /usr/lib/x86_64-linux-gnu/libz.so
toy: /usr/lib/x86_64-linux-gnu/libtinfo.so
toy: /usr/lib/llvm-14/lib/libLLVMDemangle.a
toy: CMakeFiles/toy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable toy"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/toy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/toy.dir/build: toy
.PHONY : CMakeFiles/toy.dir/build

CMakeFiles/toy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/toy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/toy.dir/clean

CMakeFiles/toy.dir/depend:
	cd /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build /mnt/d/compilers/llvm14_tuts/solutions/ch4JIT/build/CMakeFiles/toy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/toy.dir/depend
