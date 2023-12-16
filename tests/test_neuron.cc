#include "neuron.h"
#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE("Test neuron", "[Neuron]")
{
    /* initialize random seed: */
    size_t n_in = 10;
    Neuron neuron(n_in);

    SECTION("Test normal inputs")
    {
        std::vector<double> input(n_in);
        for (std::vector<double>::size_type i = 0; i < n_in; i++)
        {
            input[i] = 1.0;
        }

        Variable result = neuron.forward(input);
        REQUIRE(result.gradient() == 0.0);
        result.set_gradient(1.0);
        REQUIRE(result.gradient() == 1.0);
        REQUIRE(result.children().size() == 1);
        const auto &child = result.children()[0];
        REQUIRE(child.gradient() == 0.0);
        REQUIRE(child.children().size() == 2);

        result.backward();
        REQUIRE(child.gradient() ==
                Approx(1.0 - result.value() * result.value()));
        const std::vector<Variable> &parameters = neuron.parameters();
        REQUIRE(parameters.size() == n_in + 1);
        for (size_t i = 0; i < n_in; i++)
        {
            REQUIRE(parameters[i].gradient() ==
                    Approx(child.gradient() * input[i]));
        }
        REQUIRE(neuron.bias().gradient() == Approx(child.gradient()));

        for (std::vector<double>::size_type i = 0; i < n_in; i++)
        {
            REQUIRE(parameters[i].gradient() ==
                    Approx(child.gradient() * input[i]));
        }
        REQUIRE(parameters[n_in].gradient() == Approx(child.gradient()));
    }

    SECTION("test variable")
    {
        std::vector<Variable> input(n_in);
        for (size_t i = 0; i < n_in; i++)
        {
            input[i] = Variable(rand() % 3 - 1.0);
            REQUIRE(input[i].reference() == &input[i]);
        }

        Variable result = neuron.forward(input);
        REQUIRE(result.gradient() == 0.0);
        result.set_gradient(1.0);
        REQUIRE(result.gradient() == 1.0);
        REQUIRE(result.children().size() == 1);
        const auto &child = result.children()[0];
        REQUIRE(child.gradient() == 0.0);
        REQUIRE(child.children().size() == 2);
        const auto &grandson = child.children()[0];
        REQUIRE(grandson.children().size() == 2 * n_in);
        result.backward();
        REQUIRE(child.gradient() ==
                Approx(1.0 - result.value() * result.value()));
        std::vector<Variable> parameters = neuron.parameters();
        REQUIRE(parameters.size() == n_in + 1);
        for (size_t i = 0; i < n_in; i++)
        {
            REQUIRE(parameters[i].reference() == &neuron.weights()[i]);
            REQUIRE(parameters[i].value() ==
                    Approx(neuron.weights()[i].value()));
            REQUIRE(parameters[i + n_in].reference() == &input[i]);
            REQUIRE(parameters[i + n_in].value() == Approx(input[i].value()));
        }
        for (size_t i = 0; i < n_in; i++)
        {
            REQUIRE(parameters[i].gradient() ==
                    Approx(child.gradient() * input[i].value()));
        }
        REQUIRE(parameters[n_in].gradient() == Approx(child.gradient()));
        for (size_t i = 0; i < n_in; i++)
        {
            REQUIRE(grandson.children()[i + n_in].reference() == &input[i]);
            REQUIRE(input[i].gradient() ==
                    Approx(child.gradient() * parameters[i].value()));
        }
    }
}
