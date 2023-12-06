cmake_minimum_required(VERSION 3.0.0)
project(furina VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include(CTest)
enable_testing()
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(furina main.cpp)
target_link_libraries(furina
    ${OpenCV_LIBS})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)