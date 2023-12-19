if(ENABLE_TESTING)
    set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/test_variable.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_variable_variable.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_variable_constant.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_constant_variable.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_math_operation.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_active_function.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_computational_graph.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_neuron.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_loss.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_mlp.cc"
        )
    set(TEST_HEADERS "")

    add_executable(${UNIT_TEST_NAME} ${TEST_SOURCES} ${TEST_HEADERS})

    target_link_libraries(${UNIT_TEST_NAME} PUBLIC ${VARIABLE} ${NEURON} ${LAYER} ${NEURAL_NETWORK} ${LOSS})
    target_link_libraries(${UNIT_TEST_NAME} PRIVATE Catch2::Catch2)

    target_set_warnings(
        TARGET
        ${UNIT_TEST_NAME}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})

    if(ENABLE_COVERAGE)
        set(COVERAGE_MAIN "coverage")
        set(COVERAGE_EXCLUDES
            "${PROJECT_SOURCE_DIR}/app/*"
            "${PROJECT_SOURCE_DIR}/cmake/*"
            "${PROJECT_SOURCE_DIR}/docs/*"
            "${PROJECT_SOURCE_DIR}/external/*"
            "${PROJECT_SOURCE_DIR}/tests/*"
            "${PROJECT_SOURCE_DIR}/build/*"
            "/usr/include/*")
        setup_target_for_coverage_lcov(
            NAME
            ${COVERAGE_MAIN}
            EXECUTABLE
            ${UNIT_TEST_NAME}
            DEPENDENCIES
            ${UNIT_TEST_NAME})
    endif()
endif()
