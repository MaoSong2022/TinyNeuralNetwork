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

        std::vector<double> inputs{1.0, -2.0, 3.0};
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

        const auto &parameters = layer.parameters();
        const auto &neuron = layer.neurons()[0];
        REQUIRE(child.children()[1].reference() == &neuron.bias());
        REQUIRE(neuron.bias().gradient() == Approx(child.gradient()));
        const auto &product = child.children()[0];
        REQUIRE(product.gradient() == Approx(child.gradient()));
        for (size_t i = 0; i < 4; ++i)
        {
            REQUIRE(parameters[i].reference() ==
                    neuron.parameters()[i].reference());
        }
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(parameters[i].reference() == &neuron.weights()[i]);
        }
        REQUIRE(parameters[3].reference() == &neuron.bias());
        REQUIRE(product.children().size() == 3);
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(product.children()[i].reference() == &neuron.weights()[i]);
        }

        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(neuron.weights()[i].gradient() ==
                    Approx(product.gradient() * inputs[i]));
        }
    }

    SECTION("Test variable inputs")
    {
        Layer layer(3, 1);
        REQUIRE(layer.neurons().size() == 1);
        REQUIRE(layer.neurons()[0].parameters().size() == 4);
        REQUIRE(layer.parameters().size() == 4);

        std::vector<Variable> inputs(3);
        for (size_t i = 0; i < 3; ++i)
        {
            inputs[i] =
                Variable(1.0 * i, 0.0, "", "input[" + std::to_string(i) + "]");
        }
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

        const auto &parameters = layer.parameters();
        const auto &neuron = layer.neurons()[0];
        REQUIRE(child.children()[1].reference() == &neuron.bias());
        REQUIRE(neuron.bias().gradient() == Approx(child.gradient()));
        const auto &product = child.children()[0];
        REQUIRE(product.gradient() == Approx(child.gradient()));
        for (size_t i = 0; i < 4; ++i)
        {
            REQUIRE(parameters[i].reference() ==
                    neuron.parameters()[i].reference());
        }
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(parameters[i].reference() == &neuron.weights()[i]);
        }
        REQUIRE(parameters[3].reference() == &neuron.bias());
        REQUIRE(product.children().size() == 6);
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(product.children()[i].reference() == &neuron.weights()[i]);
            REQUIRE(product.children()[i + 3].reference() == &inputs[i]);
        }

        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(neuron.weights()[i].gradient() ==
                    Approx(product.gradient() * inputs[i].value()));
        }
    }
}
