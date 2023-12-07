#include "variable.h"
#include <catch2/catch.hpp>


TEST_CASE("Test active functions", "[Variable-Active-Function]")
{
    Variable a(2.0);

    SECTION("Test ReLU case 1")
    {
        Variable b = a.relu();
        REQUIRE(b.value() == 2.0);
        REQUIRE(b.children().size() == 1);
        b.set_gradient(1.0);
        b.backward();
        REQUIRE(b.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
    }

    SECTION("Test ReLU case 2")
    {
        Variable temp(-1.0);
        Variable c = temp.relu();
        REQUIRE(c.value() == 0.0);
        REQUIRE(c.children().size() == 1);
        c.set_gradient(1.0);
        c.backward();
        REQUIRE(c.gradient() == 1.0);
        REQUIRE(temp.gradient() == 0.0);
    }

    SECTION("Test tanh")
    {
        Variable d = a.tanh();
        REQUIRE(d.value() == 0.9640275800758169);
        REQUIRE(d.children().size() == 1);
        d.set_gradient(1.0);
        d.backward();
        REQUIRE(d.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.07065082485316443);
    }

    SECTION("Test sigmoid")
    {
        Variable e = a.sigmoid();
        REQUIRE(e.value() == 0.8807970779778823);
        REQUIRE(e.children().size() == 1);
        e.set_gradient(1.0);
        e.backward();
        REQUIRE(e.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.10499358540350662);
    }
}
