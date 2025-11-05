cmake_minimum_required(VERSION 3.15)

project(transformer
    VERSION 0.1.0
    DESCRIPTION "Transformer project"
    LANGUAGES CXX
)

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# 示例：链接线程库（根据需要查找其他依赖）
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)

# 添加可执行文件
add_executable(${PROJECT_NAME}_app
    src/main.cpp
)

# 链接库到可执行文件
target_link_libraries(${PROJECT_NAME}_app
    PRIVATE
        ${PROJECT_NAME}
)

# 可执行文件也需要包含目录（如果直接包含头文件）
target_include_directories(${PROJECT_NAME}_app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)