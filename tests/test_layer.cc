#include "layer.h"
#include <catch2/catch.hpp>

TEST_CASE("Test layer", "[Layer]")
{
    SECTION("Test normal inputs")
    {
        Layer layer(3, 1);
        REQUIRE(layer.neurons().size() == 1);
        REQUIRE(layer.neurons()[0].parameters().size() == 4);
        REQUIRE(layer.parameters().size() == 4);

        std::vector<double> inputs{1.0, 2.0, 3.0};
        std::vector<Variable> results = layer.forward(inputs);
        REQUIRE(results.size() == 1);
        auto &result = results[0];
        REQUIRE(result.children().size() == 1);
        result.set_gradient(1.0);
        result.backward();
        REQUIRE(result.gradient() == Approx(1.0));

        auto &child = result.children()[0];
        REQUIRE(child.children().size() == 2);
        REQUIRE(child.gradient() ==
                Approx(1.0 - result.value() * result.value()));
        REQUIRE(layer.parameters()[0].gradient() ==
                Approx(child.gradient() * inputs[0]));
        REQUIRE(layer.parameters()[1].gradient() ==
                Approx(child.gradient() * inputs[1]));
        REQUIRE(layer.parameters()[2].gradient() ==
                Approx(child.gradient() * inputs[2]));
        REQUIRE(layer.parameters()[3].gradient() == Approx(child.gradient()));
    }
}
