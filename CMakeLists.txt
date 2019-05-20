cmake_minimum_required(VERSION 3.9)
project(test VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)

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

#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/repo/CMakeLists.txt")
#    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
#endif()
add_subdirectory(docs)

add_subdirectory(extern/Catch2)
include_directories(${CATCH_INCLUDE_DIRS})

add_subdirectory(extern/eigen-git-mirror)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "##############################################${EIGEN3_INCLUDE_DIR}aaaaaaaaaa###############")
add_subdirectory(extern/pybind11)
add_subdirectory(extern/spdlog)

#https://github.com/StableCoder/cmake-scripts
include(code-coverage.cmake)
include(sanitizers.cmake)

#add_subdirectory(src)
set(SRC_FILES src/test.cpp src/graph_node.cpp include/libdl/opperation_node.h include/libdl/Constant.h include/libdl/Variable.h include/libdl/opperation.h include/libdl/placeholder.h include/libdl/graph.h src/Constant.cpp src/Variable.cpp src/opperation.cpp src/placeholder.cpp src/graph.cpp include/libdl/Optimizer.h src/Optimizer.cpp include/libdl/loss.h src/loss.cpp include/libdl/initializer.h src/initializer.cpp)


add_library(${MY_LIB_NAME} SHARED ${SRC_FILES})
target_link_libraries(${MY_LIB_NAME} PRIVATE eigen)
target_include_directories(${MY_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${MY_LIB_NAME} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})


find_program (CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found!")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-format-style=file;-header-filter='${CMAKE_SOURCE_DIR}/*'" CACHE STRING "" FORCE)
else()
    message(STATUS "clang-tidy NOT found!")
    set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE) # clear it
endif()

set_target_properties(${MY_LIB_NAME} PROPERTIES CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")


set(TEST_SRC_FILES  tests/mainTest.cpp tests/trytest.cpp tests/test_graph_node.cpp tests/test_opperation_node.cpp tests/test_variable.cpp tests/test_constant.cpp tests/test_opperation.cpp tests/test_placeholder.cpp tests/test_graph.cpp tests/test_Optimizer.cpp tests/test_loss.cpp tests/test_initializer.cpp)
add_executable(my_tests ${TEST_SRC_FILES})
target_link_libraries(my_tests PRIVATE Catch2)
target_link_libraries(my_tests PRIVATE eigen)

target_link_libraries(my_tests PRIVATE ${MY_LIB_NAME})

set(TARGET_NAME my_tests)

target_code_coverage(my_tests)

if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    #add_custom_target(${TARGET_NAME}-run ALL COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endif()

if( ${CMAKE_BUILD_TYPE} STREQUAL "Release" )
endif()


add_executable(XOR xor_problem.cpp)
target_link_libraries(XOR PRIVATE eigen)
target_link_libraries(XOR PRIVATE ${MY_LIB_NAME})
