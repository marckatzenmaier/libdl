#set(SRC_FILES test.cpp)


#add_library(${MY_LIB_NAME} SHARED ${SRC_FILES})
#target_include_directories(${MY_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
#set_target_properties(${MY_LIB_NAME} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

#pybind11_add_module(pythonBinding libdl)
#set(MY_PY_LIB_NAME ${MY_LIB_NAME}_py)
#add_library(${MY_PY_LIB_NAME} SHARED ${SRC_FILES})
#target_include_directories(${MY_PY_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
#target_link_libraries(${MY_PY_LIB_NAME} PRIVATE pybind11::module)

#set_target_properties(${MY_PY_LIB_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
#        SUFFIX "${PYTHON_MODULE_EXTENSION}" OUTPUT_NAME ${MY_LIB_NAME} LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})