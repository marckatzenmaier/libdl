cmake_minimum_required(VERSION 3.9)
project(test VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MY_LIB_NAME my_dllib)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

#precompiled header stuff
#set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
#include(cotire.cmake)

add_subdirectory(docs)

set(CATCH_INCLUDE_DIRS extern/Catch2/single_include)
include_directories(${CATCH_INCLUDE_DIRS})

set(EIGEN3_INCLUDE_DIR extern/eigen-git-mirror)
include_directories(${EIGEN3_INCLUDE_DIR})

set(PYBIND11_CPP_STANDARD -std=c++1z)
add_subdirectory(extern/pybind11)

#https://github.com/StableCoder/cmake-scripts
include(code-coverage.cmake)

#add_subdirectory(src)
set(SRC_FILES src/test.cpp src/graph_node.cpp include/libdl/Variable.h include/libdl/opperation.h include/libdl/placeholder.h include/libdl/graph.h src/Variable.cpp src/opperation.cpp src/placeholder.cpp src/graph.cpp include/libdl/Optimizer.h src/Optimizer.cpp include/libdl/loss.h src/loss.cpp include/libdl/initializer.h src/initializer.cpp src/math_functions.cpp include/libdl/math_functions.h include/libdl/helper_functions.h src/helper_functions.cpp)


add_library(${MY_LIB_NAME} SHARED ${SRC_FILES})
target_include_directories(${MY_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${MY_LIB_NAME} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
#cotire(${MY_LIB_NAME})


find_program (CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found!")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-format-style=file;-header-filter='${CMAKE_SOURCE_DIR}/*'" CACHE STRING "" FORCE)
else()
    message(STATUS "clang-tidy NOT found!")
    set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE) # clear it
endif()

set_target_properties(${MY_LIB_NAME} PROPERTIES CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")


set(TEST_SRC_FILES  tests/mainTest.cpp tests/trytest.cpp tests/test_graph_node.cpp tests/test_variable.cpp tests/test_opperation.cpp tests/test_placeholder.cpp tests/test_graph.cpp tests/test_Optimizer.cpp tests/test_loss.cpp tests/test_initializer.cpp tests/test_math_functions.cpp)
add_executable(my_tests ${TEST_SRC_FILES})
target_link_libraries(my_tests PRIVATE ${MY_LIB_NAME})
#cotire(my_tests)
target_code_coverage(my_tests)


add_executable(XOR xor_problem.cpp)
target_link_libraries(XOR PRIVATE ${MY_LIB_NAME})

add_executable(MNIST mnist_problem.cpp)
target_link_libraries(MNIST PRIVATE ${MY_LIB_NAME})


#add_executable(try_stuff profile_and_try_stuff_main.cpp)
#target_link_libraries(try_stuff PRIVATE ${MY_LIB_NAME})


set(MY_PY_LIB_NAME ${MY_LIB_NAME}_py)
pybind11_add_module(${MY_PY_LIB_NAME} src/py_binding.cpp)
target_include_directories(${MY_PY_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${MY_PY_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind/include)
target_link_libraries(${MY_PY_LIB_NAME} PUBLIC ${MY_LIB_NAME})
