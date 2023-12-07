#include "variable.h"
#include <catch2/catch.hpp>
#include <math.h>

TEST_CASE("Test math operations", "[Variable-Math]")
{
    Variable a(2.0);

    SECTION("Test exponential")
    {
        Variable b = a.exp();
        REQUIRE(b.value() == 7.38905609893065);
        REQUIRE(b.children().size() == 1);
        b.set_gradient(1.0);
        b.backward();
        REQUIRE(b.gradient() == 1.0);
        REQUIRE(a.gradient() == 7.38905609893065);
    }

    SECTION("Test log")
    {
        Variable c = a.log();
        REQUIRE(c.value() == 0.6931471805599453);
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
        REQUIRE(d.value() == 0.9092974268256817);
        REQUIRE(d.children().size() == 1);
        d.set_gradient(1.0);
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == -0.4161468365471424);
    }

    SECTION("Test cos")
    {
        Variable e = a.cos();
        REQUIRE(e.value() == -0.4161468365471424);
        REQUIRE(e.children().size() == 1);
        e.set_gradient(1.0);
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == -0.9092974268256817);
    }

    SECTION("Test tan")
    {
        Variable f = a.tan();
        REQUIRE(f.value() == -2.185039863261519);
        REQUIRE(f.children().size() == 1);
        f.set_gradient(1.0);
        f.backward();
        REQUIRE(f.gradient() == 1.0);
        REQUIRE(a.gradient() == 5.774399204041917);
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
        REQUIRE(h.value() == 3.6268604078470186);
        REQUIRE(h.children().size() == 1);
        h.set_gradient(1.0);
        h.backward();
        REQUIRE(h.gradient() == 1.0);
        REQUIRE(a.gradient() == 3.7621956910836314);
    }

    SECTION("Test cosh")
    {
        Variable i = a.cosh();
        REQUIRE(i.value() == 3.7621956910836314);
        REQUIRE(i.children().size() == 1);
        i.set_gradient(1.0);
        i.backward();
        REQUIRE(i.gradient() == 1.0);
        REQUIRE(a.gradient() == 3.6268604078470186);
    }
}
