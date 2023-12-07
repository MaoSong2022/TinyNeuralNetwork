#include "variable.h"
#include <catch2/catch.hpp>


TEST_CASE("Test variable-constant operations", "[Variable-Constant]")
{
    Variable a(2.0);

    SECTION("Test variable double addition")
    {
        Variable b = a + 2.0;
        REQUIRE(b.value() == 4.0);
        REQUIRE(b.children().size() == 1);
        b.set_gradient(1.0);
        b.backward();
        REQUIRE(b.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
    }

    SECTION("Test variable double subtraction")
    {
        Variable c = a - 2.0;
        REQUIRE(c.value() == 0.0);
        REQUIRE(c.children().size() == 1);
        c.set_gradient(1.0);
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
    }

    SECTION("Test variable double multiplication")
    {
        Variable d = a * 2.0;
        REQUIRE(d.value() == 4.0);
        REQUIRE(d.children().size() == 1);
        d.set_gradient(1.0);
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == 2.0);
    }

    SECTION("Test variable double division")
    {
        Variable e = a / 2.0;
        REQUIRE(e.value() == 1.0);
        REQUIRE(e.children().size() == 1);
        e.set_gradient(1.0);
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.5);
    }

    SECTION("Test double division by zero")
    {
        REQUIRE_THROWS_AS(a / 0.0, std::overflow_error);
    }
}
