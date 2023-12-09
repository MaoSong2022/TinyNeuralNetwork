# Sources and Headers
set(LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mlp.cc")
set(LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/mlp.h")
set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")

# MyLib Library
add_library(${NEURAL_NETWORK} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_include_directories(${NEURAL_NETWORK} PUBLIC ${LIBRARY_INCLUDES})
target_link_libraries(
    ${NEURAL_NETWORK}
    PRIVATE nlohmann_json::nlohmann_json
            fmt::fmt
            spdlog::spdlog
            cxxopts::cxxopts)

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${NEURAL_NETWORK}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${NEURAL_NETWORK}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${NEURAL_NETWORK})
endif()
