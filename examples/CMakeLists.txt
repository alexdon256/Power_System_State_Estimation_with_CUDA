cmake_minimum_required(VERSION 3.18)

# Basic example
add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example PRIVATE SLE)
# CMAKE_SOURCE_DIR always points to the root CMakeLists.txt directory
target_include_directories(basic_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Real-time example
add_executable(realtime_example realtime_example.cpp)
target_link_libraries(realtime_example PRIVATE SLE)
target_include_directories(realtime_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Observability example
add_executable(observability_example observability_example.cpp)
target_link_libraries(observability_example PRIVATE SLE)
target_include_directories(observability_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Advanced features example
add_executable(advanced_features_example advanced_features_example.cpp)
target_link_libraries(advanced_features_example PRIVATE SLE)
target_include_directories(advanced_features_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Hybrid robust estimation example
add_executable(hybrid_robust_example hybrid_robust_example.cpp)
target_link_libraries(hybrid_robust_example PRIVATE SLE)
target_include_directories(hybrid_robust_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Set output directory
set_target_properties(basic_example realtime_example observability_example advanced_features_example hybrid_robust_example
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

