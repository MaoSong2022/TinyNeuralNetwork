# Sources and Headers
set(LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/neuron.cc")
set(LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/neuron.h")
set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")

# MyLib Library
add_library(${NEURON} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_include_directories(${NEURON} PUBLIC ${LIBRARY_INCLUDES})
target_link_libraries(
    ${NEURON}
    PRIVATE nlohmann_json::nlohmann_json
            fmt::fmt
            spdlog::spdlog
            cxxopts::cxxopts)

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${NEURON}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${NEURON}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${NEURON})
endif()
