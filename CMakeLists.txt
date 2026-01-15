cmake_minimum_required(VERSION 3.22)
project(relorbit_engine LANGUAGES CXX)

# CMake moderno + pybind11 em CMake >= 4 tende a reclamar; isso silencia corretamente
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Faz o pybind11 usar FindPython moderno (resolve warning e ajuda o VSCode)
set(PYBIND11_FINDPYTHON ON)

# Encontra Python do ambiente que está rodando o build
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Encontra pybind11 instalado via pip (scikit-build-core injeta CMAKE_PREFIX_PATH)
find_package(pybind11 CONFIG REQUIRED)

# Módulo Python (será importado como relorbit_py._engine)
pybind11_add_module(_engine
  src_cpp/bindings/pybind_module.cpp
  src_cpp/lib/api.cpp
)

# Seus headers
target_include_directories(_engine PRIVATE
  ${Python_INCLUDE_DIRS}
  src_cpp/include
)

# Flags recomendadas
if (MSVC)
  target_compile_options(_engine PRIVATE /EHsc /O2)
else()
  target_compile_options(_engine PRIVATE -O3)
endif()

set_target_properties(_engine PROPERTIES OUTPUT_NAME "_engine")

# Instala o módulo dentro do pacote Python
install(TARGETS _engine
  LIBRARY DESTINATION relorbit_py
  RUNTIME DESTINATION relorbit_py
  ARCHIVE DESTINATION relorbit_py
)
