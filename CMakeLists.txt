cmake_minimum_required(VERSION 3.9)
project(test VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)

set(MY_LIB_NAME my_dllib)

add_subdirectory(docs)

add_subdirectory(extern/Catch2)
include_directories(${CATCH_INCLUDE_DIRS})

add_subdirectory(extern/eigen-git-mirror)
add_subdirectory(extern/pybind11)
add_subdirectory(extern/spdlog)

#https://github.com/StableCoder/cmake-scripts
include(code-coverage.cmake)
include(sanitizers.cmake)

#add_subdirectory(src)
set(SRC_FILES src/test.cpp)


add_library(${MY_LIB_NAME} SHARED ${SRC_FILES})
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



add_executable(my_tests tests/mainTest.cpp tests/trytest.cpp)
target_link_libraries(my_tests PRIVATE Catch2)
target_link_libraries(my_tests PRIVATE ${MY_LIB_NAME})

set(TARGET_NAME my_tests)

target_code_coverage(my_tests)

if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    #add_custom_target(${TARGET_NAME}-run ALL COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endif()

if( ${CMAKE_BUILD_TYPE} STREQUAL "Release" )
endif()
