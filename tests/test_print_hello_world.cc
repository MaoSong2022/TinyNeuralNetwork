#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "my_lib.h"


TEST_CASE("Test printer function", "[print_hello_world]")
{
    REQUIRE(print_hello_world() == 1);
}
