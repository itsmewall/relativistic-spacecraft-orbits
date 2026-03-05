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

# ── Eigen3 ────────────────────────────────────────────────────────
# Procura Eigen instalado no sistema; se não encontrar, usa o bundled.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package — using bundled third_party/eigen")
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
        message(FATAL_ERROR "Eigen not found at ${EIGEN3_INCLUDE_DIR}. "
            "Run: git submodule update --init --recursive  or "
            "place Eigen headers in third_party/eigen/")
    endif()
    # Cria target de interface compatível com o target Eigen3::Eigen
    add_library(Eigen3_bundled INTERFACE)
    add_library(Eigen3::Eigen ALIAS Eigen3_bundled)
    target_include_directories(Eigen3_bundled INTERFACE "${EIGEN3_INCLUDE_DIR}")
endif()

pybind11_add_module(_engine
  src_cpp/bindings/pybind_module.cpp
  src_cpp/lib/api.cpp
  src_cpp/lib/schwarzschild_equatorial.cpp
  src_cpp/lib/kerr_equatorial.cpp
  src_cpp/lib/schwarzschild_lowthrust.cpp
  src_cpp/lib/kerr_lowthrust.cpp
  src_cpp/lib/attitude.cpp
  src_cpp/lib/schwarzschild_6dof.cpp
  src_cpp/include/relorbit/types.hpp
  src_cpp/include/relorbit/api.hpp
  src_cpp/include/relorbit/models/schwarzschild_equatorial.hpp
  src_cpp/include/relorbit/models/schwarzschild_lowthrust.hpp
  src_cpp/include/relorbit/models/kerr_lowthrust.hpp
  src_cpp/include/relorbit/models/attitude.hpp
  src_cpp/include/relorbit/models/schwarzschild_6dof.hpp
  src_cpp/lib/gr/kerr_metric.cpp
  src_cpp/lib/kerr_6dof.cpp
  src_cpp/lib/gr/kerr_null_geodesic.cpp
)

target_include_directories(_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include
)
target_link_libraries(_engine PRIVATE Eigen3::Eigen)

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