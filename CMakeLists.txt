cmake_minimum_required(VERSION 3.15)

project(transformer
    VERSION 0.1.0
    DESCRIPTION "Transformer project"
    LANGUAGES CXX
)

# option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
# option(BUILD_EXAMPLES "Build examples" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(xtl REQUIRED)
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${xtl_INCLUDE_DIRS}) 
include_directories(${xsimd_INCLUDE_DIRS}) 
include_directories(${xtensor_INCLUDE_DIRS}) 
include_directories(${spdlog_INCLUDE_DIRS})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 源文件收集
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS 
    src/*.cpp 
    src/*.c
)

# 头文件收集（可选，仅用于IDE显示）
file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS 
    include/*.hpp 
    include/*.h
)

# 库/目标
add_library(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# 编译选项按目标设置
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native -DNDEBUG)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native -DNDEBUG)
endif()

# 示例：链接线程库（根据需要查找其他依赖）
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# 添加可执行文件
add_executable(${PROJECT_NAME}_app
    src/main.cpp
)

# 查找 BLAS（尽量使用系统或包管理器的实现）
find_package(BLAS QUIET)

if(BLAS_FOUND)
  message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
else()
  if(APPLE)
    message(STATUS "BLAS not found via find_package — will link Accelerate on macOS")
  else()
    message(WARNING "BLAS not found. Consider installing OpenBLAS or MKL.")
  endif()
endif()

# 链接库到可执行文件，统一处理 BLAS/Accelerate
target_link_libraries(${PROJECT_NAME}_app
  PRIVATE
    ${PROJECT_NAME}
    xtl
    xsimd
    xtensor
    Threads::Threads
    fmt::fmt
)

if(BLAS_FOUND)
  target_link_libraries(${PROJECT_NAME}_app PRIVATE ${BLAS_LIBRARIES})
elseif(APPLE)
  target_link_libraries(${PROJECT_NAME}_app PRIVATE "-framework Accelerate")
else()
  message(FATAL_ERROR "No BLAS available. Install OpenBLAS or set BLAS_LIBRARIES.")
endif()

# 可执行文件也需要包含本地 include（库的 PUBLIC include 已经传播给可执行）
target_include_directories(${PROJECT_NAME}_app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 添加 tests 支持（编译 tests/*.cpp 并注册为 CTest 测试）
option(BUILD_TESTS "Build tests" ON)
include(CTest)
if(BUILD_TESTS)
  enable_testing()

  # 查找所有 C++ 测试源文件
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

  foreach(_test_src IN LISTS TEST_SOURCES)
    get_filename_component(_test_name ${_test_src} NAME_WE)
    set(_exe_name test_${_test_name})

    add_executable(${_exe_name} ${_test_src})

    # 链接到主库和第三方（与可执行相同的依赖）
    target_link_libraries(${_exe_name}
      PRIVATE
        ${PROJECT_NAME}
        xtl
        xsimd
        xtensor
        Threads::Threads
        fmt::fmt
    )

    # 可选：在 macOS 上为需要 BLAS 的测试添加 Accelerate，非 macOS 可选链接 BLAS
    if(APPLE)
      target_link_libraries(${_exe_name} PRIVATE "-framework Accelerate")
    else()
      find_package(BLAS QUIET)
      if(BLAS_FOUND)
        target_link_libraries(${_exe_name} PRIVATE ${BLAS_LIBRARIES})
      endif()
    endif()

    target_include_directories(${_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(${_exe_name} PRIVATE cxx_std_20)

    # 注册到 CTest
    add_test(NAME ${_test_name} COMMAND $<TARGET_FILE:${_exe_name}>)
    set_tests_properties(${_test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
endif()