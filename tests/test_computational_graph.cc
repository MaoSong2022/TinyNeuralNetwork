#include "variable.h"
#include <catch2/catch.hpp>


TEST_CASE("Test computational graph", "[Computation Graph]")
{
    Variable x1(2.0);
    Variable x2(3.0);

    Variable f = x1.log() + x1 * x2 + x2.sin();

    REQUIRE(f.value() == 6.834267188619813);
    REQUIRE(f.children().size() == 2);
    f.set_gradient(1.0);
    f.backward();
    REQUIRE(f.gradient() == 1.0);
    REQUIRE(x1.gradient() == 3.5);
    REQUIRE(x2.gradient() == 1.0100075033995546);
}
