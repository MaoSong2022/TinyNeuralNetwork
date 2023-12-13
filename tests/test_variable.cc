#define CATCH_CONFIG_MAIN
#include "variable.h"
#include <catch2/catch.hpp>


TEST_CASE("Test variable basic", "[Variable]")
{
    Variable a(2.0, 0.0, "", "a");

    SECTION("Test copy constructor")
    {
        Variable b(a);
        REQUIRE(b.value() == 2.0);
        REQUIRE(b.gradient() == 0.0);
        REQUIRE(b.op() == "");
        REQUIRE(b.name() == "a");
        REQUIRE(b.children().size() == 0);
        REQUIRE(b.ref == &a);
    }

    SECTION("Test move constructor")
    {
        Variable b(Variable(2.0, 0.0, "", "b"));
        REQUIRE(b.value() == 2.0);
        REQUIRE(b.gradient() == 0.0);
        REQUIRE(b.op() == "");
        REQUIRE(b.name() == "b");
        REQUIRE(b.children().size() == 0);
        REQUIRE(b.ref == &b);
    }

    SECTION("Test copy assignment")
    {
        Variable b(2.0, 0.0, "", "b");
        b = a;
        REQUIRE(b.value() == 2.0);
        REQUIRE(b.gradient() == 0.0);
        REQUIRE(b.op() == "");
        REQUIRE(b.name() == "a");
        REQUIRE(b.children().size() == 0);
        REQUIRE(b.ref == &a);
    }

    SECTION("Test move assignment")
    {
        Variable b(2.0, 0.0, "", "b");
        b = Variable(2.0, 0.0, "", "b");
        REQUIRE(b.value() == 2.0);
        REQUIRE(b.gradient() == 0.0);
        REQUIRE(b.op() == "");
        REQUIRE(b.name() == "b");
        REQUIRE(b.children().size() == 0);
        REQUIRE(b.ref == &b);
    }

    SECTION("Test zero gradient")
    {
        REQUIRE(a.gradient() == 0.0);
        a.set_gradient(1.0);
        REQUIRE(a.gradient() == 1.0);
        a.zero_gradient();
        REQUIRE(a.gradient() == 0.0);
    }

    SECTION("Test gradient descent")
    {
        a.set_value(2.0);
        REQUIRE(a.value() == 2.0);
        a.set_gradient(1.0);
        REQUIRE(a.gradient() == 1.0);
        a.gradient_descent(0.1);
        REQUIRE(a.value() == Approx(1.9));
        a.gradient_descent(1.0);
        REQUIRE(a.value() == Approx(0.9));
    }
}
