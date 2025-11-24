cmake_minimum_required(VERSION 3.18)

# Consolidated setup examples
add_executable(offlinesetup offlinesetup.cpp)
target_link_libraries(offlinesetup PRIVATE SLE)
target_include_directories(offlinesetup PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(realtimesetup realtimesetup.cpp)
target_link_libraries(realtimesetup PRIVATE SLE)
target_include_directories(realtimesetup PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(hybridsetup hybridsetup.cpp)
target_link_libraries(hybridsetup PRIVATE SLE)
target_include_directories(hybridsetup PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(advancedsetup advancedsetup.cpp)
target_link_libraries(advancedsetup PRIVATE SLE)
target_include_directories(advancedsetup PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(compare_measured_estimated compare_measured_estimated.cpp)
target_link_libraries(compare_measured_estimated PRIVATE SLE)
target_include_directories(compare_measured_estimated PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Set output directory
set_target_properties(offlinesetup realtimesetup hybridsetup advancedsetup compare_measured_estimated
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

