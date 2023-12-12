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

    SECTION("Test activate tanh")
    {
        Variable f = a.activate("tanh");
        REQUIRE(f.value() == 0.9640275800758169);
        REQUIRE(f.children().size() == 1);
        f.set_gradient(1.0);
        f.backward();
        REQUIRE(f.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.07065082485316443);
    }

    SECTION("Test activate sigmoid")
    {
        Variable g = a.activate("sigmoid");
        REQUIRE(g.value() == 0.8807970779778823);
        REQUIRE(g.children().size() == 1);
        g.set_gradient(1.0);
        g.backward();
        REQUIRE(g.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.10499358540350662);
    }

    SECTION("Test activate relu")
    {
        Variable h = a.activate("relu");
        REQUIRE(h.value() == 2.0);
        REQUIRE(h.children().size() == 1);
        h.set_gradient(1.0);
        h.backward();
        REQUIRE(h.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
    }

    SECTION("Test activate identity")
    {
        Variable h = a.activate("identity");
        REQUIRE(h.value() == 2.0);
        REQUIRE(h.children().size() == 1);
        h.set_gradient(1.0);
        h.backward();
        REQUIRE(h.gradient() == 1.0);
        REQUIRE(a.gradient() == 1.0);
    }

    SECTION("Test composition")
    {
        Variable i = a.tanh().tanh();
        REQUIRE(i.value() == Approx(0.7460679984455996));
        REQUIRE(i.children().size() == 1);
        i.set_gradient(1.0);
        i.backward();
        REQUIRE(i.gradient() == 1.0);
        REQUIRE(a.gradient() == 0.031325342296270944);
    }
}
