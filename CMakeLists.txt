cmake_minimum_required(VERSION 3.22)
project(relorbit_engine LANGUAGES CXX)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_engine
  src_cpp/bindings/pybind_module.cpp
  src_cpp/lib/api.cpp
  src_cpp/lib/schwarzschild_equatorial.cpp

  # (Opcional, mas bom pro IDE)
  src_cpp/include/relorbit/types.hpp
  src_cpp/include/relorbit/api.hpp
  src_cpp/include/relorbit/models/schwarzschild_equatorial.hpp
)

target_include_directories(_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include
)

set_target_properties(_engine PROPERTIES
  OUTPUT_NAME "_engine"
  POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
  target_compile_options(_engine PRIVATE /EHsc /O2)
else()
  target_compile_options(_engine PRIVATE -O3)
endif()

install(TARGETS _engine
  LIBRARY DESTINATION relorbit_py
  RUNTIME DESTINATION relorbit_py
  ARCHIVE DESTINATION relorbit_py
)
