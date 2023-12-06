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
CMAKE_SOURCE_DIR = /home/caiersheng/下载/furina

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/caiersheng/下载/furina/build

# Include any dependencies generated for this target.
include CMakeFiles/opencv_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencv_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv_test.dir/flags.make

CMakeFiles/opencv_test.dir/main.cpp.o: CMakeFiles/opencv_test.dir/flags.make
CMakeFiles/opencv_test.dir/main.cpp.o: ../main.cpp
CMakeFiles/opencv_test.dir/main.cpp.o: CMakeFiles/opencv_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caiersheng/下载/furina/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv_test.dir/main.cpp.o"
	/usr/bin/clang++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv_test.dir/main.cpp.o -MF CMakeFiles/opencv_test.dir/main.cpp.o.d -o CMakeFiles/opencv_test.dir/main.cpp.o -c /home/caiersheng/下载/furina/main.cpp

CMakeFiles/opencv_test.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test.dir/main.cpp.i"
	/usr/bin/clang++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caiersheng/下载/furina/main.cpp > CMakeFiles/opencv_test.dir/main.cpp.i

CMakeFiles/opencv_test.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test.dir/main.cpp.s"
	/usr/bin/clang++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caiersheng/下载/furina/main.cpp -o CMakeFiles/opencv_test.dir/main.cpp.s

# Object files for target opencv_test
opencv_test_OBJECTS = \
"CMakeFiles/opencv_test.dir/main.cpp.o"

# External object files for target opencv_test
opencv_test_EXTERNAL_OBJECTS =

opencv_test: CMakeFiles/opencv_test.dir/main.cpp.o
opencv_test: CMakeFiles/opencv_test.dir/build.make
opencv_test: /usr/local/lib/libopencv_dnn.so.3.4.15
opencv_test: /usr/local/lib/libopencv_highgui.so.3.4.15
opencv_test: /usr/local/lib/libopencv_ml.so.3.4.15
opencv_test: /usr/local/lib/libopencv_objdetect.so.3.4.15
opencv_test: /usr/local/lib/libopencv_shape.so.3.4.15
opencv_test: /usr/local/lib/libopencv_stitching.so.3.4.15
opencv_test: /usr/local/lib/libopencv_superres.so.3.4.15
opencv_test: /usr/local/lib/libopencv_videostab.so.3.4.15
opencv_test: /usr/local/lib/libopencv_calib3d.so.3.4.15
opencv_test: /usr/local/lib/libopencv_features2d.so.3.4.15
opencv_test: /usr/local/lib/libopencv_flann.so.3.4.15
opencv_test: /usr/local/lib/libopencv_photo.so.3.4.15
opencv_test: /usr/local/lib/libopencv_video.so.3.4.15
opencv_test: /usr/local/lib/libopencv_videoio.so.3.4.15
opencv_test: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
opencv_test: /usr/local/lib/libopencv_imgproc.so.3.4.15
opencv_test: /usr/local/lib/libopencv_core.so.3.4.15
opencv_test: CMakeFiles/opencv_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/caiersheng/下载/furina/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencv_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv_test.dir/build: opencv_test
.PHONY : CMakeFiles/opencv_test.dir/build

CMakeFiles/opencv_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv_test.dir/clean

CMakeFiles/opencv_test.dir/depend:
	cd /home/caiersheng/下载/furina/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/caiersheng/下载/furina /home/caiersheng/下载/furina /home/caiersheng/下载/furina/build /home/caiersheng/下载/furina/build /home/caiersheng/下载/furina/build/CMakeFiles/opencv_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencv_test.dir/depend

