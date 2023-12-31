#include "variable.h"
#include <catch2/catch.hpp>
#include <math.h>

TEST_CASE("Test math operations", "[Variable-Math]")
{
    Variable a(2.0);

    SECTION("Test exponential")
    {
        Variable b = a.exp();
        REQUIRE(b.value() == Approx(7.3890560989));
        REQUIRE(b.children().size() == 1);
        b.set_gradient(1.0);
        b.backward();
        REQUIRE(b.gradient() == 1.0);
        REQUIRE(a.gradient() == Approx(7.38905609));
    }

    SECTION("Test log")
    {
        Variable c = a.log();
        REQUIRE(c.value() == Approx(0.693147180559));
        REQUIRE(c.children().size() == 1);
        c.set_gradient(1.0);
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.5);
    }

    SECTION("Test log by zero")
    {
        Variable temp(0.0);
        REQUIRE_THROWS_AS(temp.log(), std::overflow_error);
    }

    SECTION("Test sin")
    {
        Variable d = a.sin();
        REQUIRE(d.value() == Approx(0.9092974268));
        REQUIRE(d.children().size() == 1);
        d.set_gradient(1.0);
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == Approx(-0.4161468365));
    }

    SECTION("Test cos")
    {
        Variable e = a.cos();
        REQUIRE(e.value() == Approx(-0.4161468365));
        REQUIRE(e.children().size() == 1);
        e.set_gradient(1.0);
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == Approx(-0.9092974268));
    }

    SECTION("Test tan")
    {
        Variable f = a.tan();
        REQUIRE(f.value() == Approx(-2.185039863261519));
        REQUIRE(f.children().size() == 1);
        f.set_gradient(1.0);
        f.backward();
        REQUIRE(f.gradient() == 1.0);
        REQUIRE(a.gradient() == Approx(5.774399204041917));
    }

    SECTION("Test tan by zero")
    {
        Variable temp(M_PI_2);
        REQUIRE_THROWS_AS(temp.tan(), std::overflow_error);
    }

    SECTION("Test pow")
    {
        Variable g = a.pow(3.0);
        REQUIRE(g.value() == 8.0);
        REQUIRE(g.children().size() == 1);
        g.set_gradient(1.0);
        g.backward();
        REQUIRE(g.gradient() == 1.0);
        REQUIRE(a.gradient() == 12.0);
    }

    SECTION("Test sinh")
    {
        Variable h = a.sinh();
        REQUIRE(h.value() == Approx(3.6268604078));
        REQUIRE(h.children().size() == 1);
        h.set_gradient(1.0);
        h.backward();
        REQUIRE(h.gradient() == 1.0);
        REQUIRE(a.gradient() == Approx(3.7621956910));
    }

    SECTION("Test cosh")
    {
        Variable i = a.cosh();
        REQUIRE(i.value() == Approx(3.7621956910));
        REQUIRE(i.children().size() == 1);
        i.set_gradient(1.0);
        i.backward();
        REQUIRE(i.gradient() == 1.0);
        REQUIRE(a.gradient() == Approx(3.6268604078));
    }
}
