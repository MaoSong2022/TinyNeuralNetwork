#include "variable.h"
#include <catch2/catch.hpp>


TEST_CASE("Test operations", "[Variable]")
{
    Variable a(2.0);
    Variable b(3.0);
    Variable c = a + b;

    SECTION("Test addition")
    {
        REQUIRE(c.value() == 5.0);

        c.mutable_gradient() = 1.0;
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
        REQUIRE(b.gradient() == 1.0);
    }

    Variable d = a - b;
    SECTION("TEST subtraction")
    {
        REQUIRE(d.value() == -1.0);

        d.mutable_gradient() = 1.0;
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
        REQUIRE(b.gradient() == -1.0);
    }

    Variable e = a * b;
    SECTION("Test multiplication")
    {
        REQUIRE(e.value() == 6.0);

        e.mutable_gradient() = 1.0;
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == 3.0);
        REQUIRE(b.gradient() == 2.0);
    }

    Variable f = a / b;
    Variable temp(0.0);
    SECTION("Test division")
    {
        REQUIRE(f.value() == 0.6666666666666666);

        f.mutable_gradient() = 1.0;
        f.backward();
        REQUIRE(f.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.3333333333333333);
        REQUIRE(b.gradient() == -0.2222222222222222);

        REQUIRE_THROWS_AS(a / temp, std::overflow_error);
    }

    Variable g = a + a;
    SECTION("Test double addition")
    {
        REQUIRE(g.value() == 4.0);

        g.mutable_gradient() = 1.0;
        g.backward();
        REQUIRE(g.gradient() == 1.0);
        REQUIRE(a.gradient() == 2.0);
    }
}
