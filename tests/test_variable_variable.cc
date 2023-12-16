#include "variable.h"
#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE("Test variable-variable operations", "[Variable-Variable]")
{
    Variable a(2.0, 0.0, "", "a");
    Variable b(3.0, 0.0, "", "b");

    SECTION("Test addition")
    {
        Variable c = a + b;
        REQUIRE(c.value() == 5.0);
        REQUIRE(c.children().size() == 2);
        c.set_gradient(1.0);
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
        REQUIRE(b.gradient() == 1.0);
    }


    SECTION("TEST subtraction")
    {
        Variable d = a - b;
        REQUIRE(d.value() == -1.0);
        REQUIRE(d.children().size() == 2);
        d.set_gradient(1.0);
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
        REQUIRE(b.gradient() == -1.0);
    }

    SECTION("Test multiplication")
    {
        Variable e = a * b;
        REQUIRE(e.value() == 6.0);
        REQUIRE(e.children().size() == 2);
        e.set_gradient(1.0);
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == 3.0);
        REQUIRE(b.gradient() == 2.0);
    }


    SECTION("Test division")
    {
        Variable f = a / b;
        REQUIRE(f.value() == 0.6666666666666666);
        REQUIRE(f.children().size() == 2);
        f.set_gradient(1.0);
        f.backward();
        REQUIRE(f.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.3333333333333333);
        REQUIRE(b.gradient() == -0.2222222222222222);
    }

    SECTION("TEST division by zero")
    {
        Variable temp(0.0);
        REQUIRE_THROWS_AS(a / temp, std::overflow_error);
    }

    SECTION("Test negation")
    {
        Variable h = -a;
        REQUIRE(h.value() == -2.0);
        REQUIRE(h.children().size() == 1);
        h.set_gradient(1.0);
        h.backward();
        REQUIRE(h.gradient() == 1.0);
        REQUIRE(a.gradient() == -1.0);
    }

    SECTION("Test dot product")
    {
        std::vector<Variable> a_vec{a, a, a};
        std::vector<Variable> b_vec{b, b, b};
        Variable c = dot_product(a_vec, b_vec);
        REQUIRE(c.value() == 18.0);
        REQUIRE(c.children().size() == 6);
        for (size_t i = 0; i < 3; ++i)
        {
            std::cout << "c[" << i << "] = " << c.children()[i] << std::endl;
            REQUIRE(c.children()[i].reference() == &a);
            REQUIRE(c.children()[i + 3].reference() == &b);
        }
        c.set_gradient(1.0);
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(a.gradient() == 9.0);
        REQUIRE(b.gradient() == 6.0);
    }
}
