#include "variable.h"
#include <catch2/catch.hpp>


TEST_CASE("Test operations", "[Variable-Constant]")
{
    Variable a(2.0);

    SECTION("Test double addition")
    {
        Variable b = a + 2.0;
        REQUIRE(b.value() == 4.0);
        REQUIRE(b.children().size() == 1);
        b.mutable_gradient() = 1.0;
        b.backward();
        REQUIRE(b.gradient() == 1.0);
        REQUIRE(a.gradient() == 2.0);
    }

    SECTION("Test double subtraction")
    {
        Variable c = a - 2.0;
        REQUIRE(c.value() == 0.0);
        REQUIRE(c.children().size() == 1);
        c.mutable_gradient() = 1.0;
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
    }

    SECTION("Test double multiplication")
    {
        Variable d = a * 2.0;
        REQUIRE(d.value() == 4.0);
        REQUIRE(d.children().size() == 1);
        d.mutable_gradient() = 1.0;
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == 2.0);
    }

    SECTION("Test double division")
    {
        Variable e = a / 2.0;
        REQUIRE(e.value() == 1.0);
        REQUIRE(e.children().size() == 1);
        e.mutable_gradient() = 1.0;
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.5);
    }

    SECTION("Test double division by zero")
    {
        REQUIRE_THROWS_AS(a / 0.0, std::overflow_error);
    }
}

TEST_CASE("Test operations", "[Constant-Variable]")
{
    Variable a(2.0);

    SECTION("Test double addition")
    {
        Variable b = 2.0 + a;
        REQUIRE(b.value() == 4.0);
        REQUIRE(b.children().size() == 1);
        b.mutable_gradient() = 1.0;
        b.backward();
        REQUIRE(b.gradient() == 1.0);
        REQUIRE(a.gradient() == 2.0);
    }

    SECTION("Test double subtraction")
    {
        Variable c = 2.0 - a;
        REQUIRE(c.value() == 0.0);
        REQUIRE(c.children().size() == 1);
        c.mutable_gradient() = 1.0;
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == -1.0);
    }

    SECTION("Test double multiplication")
    {
        Variable d = 2.0 * a;
        REQUIRE(d.value() == 4.0);
        REQUIRE(d.children().size() == 1);
        d.mutable_gradient() = 1.0;
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == 2.0);
    }

    SECTION("Test double division")
    {
        Variable e = 2.0 / a;
        REQUIRE(e.value() == 1.0);
        REQUIRE(e.children().size() == 1);
        e.mutable_gradient() = 1.0;
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.5);
    }

    SECTION("Test double division by zero")
    {
        Variable temp(0.0);
        REQUIRE_THROWS_AS(2.0 / temp, std::overflow_error);
    }
}
