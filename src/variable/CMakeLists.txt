# Sources and Headers
set(LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/variable.cc")
set(LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/variable.h")
set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")

# MyLib Library
add_library(${VARIABLE} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_include_directories(${VARIABLE} PUBLIC ${LIBRARY_INCLUDES})
target_link_libraries(
    ${VARIABLE}
    PRIVATE nlohmann_json::nlohmann_json
            fmt::fmt
            spdlog::spdlog
            cxxopts::cxxopts)

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${VARIABLE}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${VARIABLE}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${VARIABLE})
endif()
