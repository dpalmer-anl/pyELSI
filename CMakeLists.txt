cmake_minimum_required(VERSION 3.20)

project(pyelsi LANGUAGES C CXX Fortran)

include(FetchContent)

# ELSI (v2.11.0) declares cmake_minimum_required(VERSION 3.0).
# CMake 4+ removed compatibility with policies <3.5 unless explicitly set.
# The variable is silently ignored on CMake <4.0 where it is not needed.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Prefer libraries from the active conda environment over system-wide libraries
# (e.g., prevents accidentally picking up MKL from a different conda env).
if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

option(PYELSI_ENABLE_MPI "Enable MPI support" ON)
option(PYELSI_ENABLE_CUDA "Enable CUDA support" OFF)
option(PYELSI_FETCH_ELSI "Fetch ELSI if not vendored" OFF)
set(PYELSI_ELSI_GIT_TAG "v2.11.0" CACHE STRING "ELSI git tag/commit to fetch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ELSI vendors older Fortran code (e.g. ELPA 2020) that can trip modern gfortran
# argument mismatch checks. This flag is required for successful builds on GCC 10+.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
endif()

# Several bundled third-party libraries (PEXSI, SCOTCH, ELPA) contain code that
# triggers new-in-GCC-14 hard errors.  Demote them back to warnings so the build
# succeeds with modern toolchains from conda-forge or system GCC 14+.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(
    # PEXSI / libstdc++ headers: bogus -Wstringop-overflow on std::fill
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-stringop-overflow>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=stringop-overflow>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  )
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  add_compile_options(
    # SCOTCH (bundled in ELSI/PEXSI) calls waitpid without including <sys/wait.h>.
    # GCC 14 makes -Wimplicit-function-declaration a hard error; demote it.
    $<$<COMPILE_LANGUAGE:C>:-Wno-implicit-function-declaration>
    # Also guard against -Wimplicit-int (same vintage of GCC 14 strictness).
    $<$<COMPILE_LANGUAGE:C>:-Wno-implicit-int>
  )
endif()

# --- Python / pybind11 / NumPy (provided by scikit-build-core) ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Threads / BLAS / LAPACK ---
find_package(Threads REQUIRED)

# We prefer linking against whatever BLAS/LAPACK is available at build time.
# For wheels, a controlled manylinux build environment should provide OpenBLAS.
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(SCALAPACK)

# Some environments (notably conda-forge toolchains) don't provide a robust
# CMake ScaLAPACK package config. We fall back to a direct library lookup.
find_library(PYELSI_SCALAPACK_LIB scalapack HINTS ENV CONDA_PREFIX PATH_SUFFIXES lib)
find_library(PYELSI_BLACS_LIB blacs HINTS ENV CONDA_PREFIX PATH_SUFFIXES lib)

# --- Optional MPI / CUDA ---
if(PYELSI_ENABLE_MPI)
  find_package(MPI REQUIRED)
else()
  message(FATAL_ERROR "pyELSI requires MPI because ELSI's C API includes mpi.h. Set -DPYELSI_ENABLE_MPI=ON.")
endif()

if(PYELSI_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# --- ELSI ---
set(_elsi_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/elsi/elsi-interface")
set(_elsi_have_vendored OFF)
if(EXISTS "${_elsi_src_dir}/CMakeLists.txt" OR EXISTS "${_elsi_src_dir}/meson.build")
  set(_elsi_have_vendored ON)
endif()

if(_elsi_have_vendored)
  message(STATUS "Using vendored ELSI at ${_elsi_src_dir}")
elseif(PYELSI_FETCH_ELSI)
  message(STATUS "Fetching ELSI from upstream (tag ${PYELSI_ELSI_GIT_TAG})")
  FetchContent_Declare(
    elsi_interface
    GIT_REPOSITORY https://gitlab.com/ElectronicStructureLibrary/elsi-interface.git
    GIT_TAG ${PYELSI_ELSI_GIT_TAG}
    GIT_SHALLOW TRUE
  )
  # Populate-only: download the source without calling add_subdirectory().
  # The explicit add_subdirectory() below is the single point of configuration,
  # which avoids the duplicate-target errors that FetchContent_MakeAvailable()
  # causes when it also calls add_subdirectory() on the same source tree.
  FetchContent_GetProperties(elsi_interface)
  if(NOT elsi_interface_POPULATED)
    FetchContent_Populate(elsi_interface)

    # ChASE in ELSI v2.11.0 uses unqualified make_unique<Matrix<T>> calls inside
    # namespace chase::mpi.  Under C++17 / GCC >= 14, std::make_unique gained a
    # different return type (std::__detail::__unique_ptr_t), making unqualified
    # lookup ambiguous.  Fully-qualify the calls — matching the fix already applied
    # in our vendored copy of this header.
    set(_chase_matrices
        "${elsi_interface_SOURCE_DIR}/external/ChASE/src/ChASE-MPI/chase_mpi_matrices.hpp")
    if(EXISTS "${_chase_matrices}")
      file(READ "${_chase_matrices}" _chase_hpp)
      string(REPLACE
          "make_unique<Matrix<T>>"
          "::chase::mpi::make_unique<Matrix<T>>"
          _chase_hpp "${_chase_hpp}")
      string(REPLACE
          "make_unique<Matrix<Base<T>>>"
          "::chase::mpi::make_unique<Matrix<Base<T>>>"
          _chase_hpp "${_chase_hpp}")
      file(WRITE "${_chase_matrices}" "${_chase_hpp}")
      message(STATUS "Patched ChASE chase_mpi_matrices.hpp for GCC 14 / C++17 compatibility")
    endif()
  endif()
  set(_elsi_src_dir "${elsi_interface_SOURCE_DIR}")
else()
  message(FATAL_ERROR "ELSI source not found. Either vendor into third_party/elsi/elsi-interface or set -DPYELSI_FETCH_ELSI=ON.")
endif()

# --- Build ELSI as a subproject ---
# ELSI is vendored/fetched into _elsi_src_dir. We build it in-tree so we can link against target `elsi`.
set(ELSI_USE_FIND_PACKAGE_MPI ON CACHE BOOL "Use find_package(MPI) for ELSI" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "Disable ELSI tests" FORCE)
set(ENABLE_C_TESTS OFF CACHE BOOL "Disable ELSI C tests" FORCE)
set(USE_ELPA_2020 OFF CACHE BOOL "Prefer newer internal ELPA when available" FORCE)

# Enable redistributed solver libraries where possible.
option(PYELSI_ENABLE_CHASE "Enable ChASE eigensolver" ON)
option(PYELSI_ENABLE_SIPS  "Enable SLEPc-SIP eigensolver (requires external SLEPc/PETSc)" OFF)

set(ENABLE_PEXSI ON CACHE BOOL "Enable PEXSI in ELSI" FORCE)
set(ENABLE_CHASE ${PYELSI_ENABLE_CHASE} CACHE BOOL "Enable ChASE in ELSI" FORCE)
set(ENABLE_SIPS  ${PYELSI_ENABLE_SIPS}  CACHE BOOL "Enable SLEPc-SIPs in ELSI" FORCE)
set(ENABLE_BSEPACK ON CACHE BOOL "Enable BSEPACK in ELSI" FORCE)
set(USE_GPU_CUDA ${PYELSI_ENABLE_CUDA} CACHE BOOL "Enable CUDA in ELSI/ELPA" FORCE)

# ELSI builds mostly STATIC libraries. They are linked into this Python extension .so;
# without -fPIC the link fails (e.g. R_X86_64_PC32 on libpexsi.a when referencing libc++).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("${_elsi_src_dir}" elsi_build)

pybind11_add_module(_pyelsi_core
  cpp/pyelsi_module.cpp
)

target_include_directories(_pyelsi_core PRIVATE cpp)
target_include_directories(_pyelsi_core PRIVATE "${_elsi_src_dir}/src/include")
if(TARGET SCALAPACK::SCALAPACK)
  target_link_libraries(_pyelsi_core PRIVATE elsi BLAS::BLAS LAPACK::LAPACK SCALAPACK::SCALAPACK)
elseif(DEFINED SCALAPACK_LIBRARIES AND NOT "${SCALAPACK_LIBRARIES}" STREQUAL "")
  target_link_libraries(_pyelsi_core PRIVATE elsi BLAS::BLAS LAPACK::LAPACK ${SCALAPACK_LIBRARIES})
elseif(PYELSI_SCALAPACK_LIB)
  target_link_libraries(_pyelsi_core PRIVATE elsi BLAS::BLAS LAPACK::LAPACK ${PYELSI_SCALAPACK_LIB})
else()
  message(FATAL_ERROR "ScaLAPACK library not found (needed for ELSI/ELPA). Install scalapack and ensure it is discoverable.")
endif()

if(PYELSI_BLACS_LIB)
  target_link_libraries(_pyelsi_core PRIVATE ${PYELSI_BLACS_LIB})
endif()

target_link_libraries(_pyelsi_core PRIVATE Threads::Threads MPI::MPI_C)

target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_MPI=1)

if(PYELSI_ENABLE_CHASE)
  target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_CHASE=1)
else()
  target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_CHASE=0)
endif()

if(PYELSI_ENABLE_SIPS)
  target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_SIPS=1)
else()
  target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_SIPS=0)
endif()

if(PYELSI_ENABLE_CUDA)
  target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_CUDA=1)
else()
  target_compile_definitions(_pyelsi_core PRIVATE PYELSI_HAS_CUDA=0)
endif()

install(TARGETS _pyelsi_core LIBRARY DESTINATION pyelsi/_core)

