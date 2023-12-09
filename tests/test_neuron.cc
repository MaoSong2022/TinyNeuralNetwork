#include "neuron.h"
#include <catch2/catch.hpp>
#include <stdlib.h>
#include <time.h>

TEST_CASE("Test neuron", "[Neuron]")
{
    /* initialize random seed: */
    srand(static_cast<unsigned int>(time(nullptr)));

    size_t n_in = 10;
    Neuron neuron(n_in);
    std::vector<double> input(n_in);
    for (std::vector<double>::size_type i = 0; i < n_in; i++)
    {
        input[i] = rand() % 3 - 1.0;
    }

    Variable result = neuron.forward(input);
    REQUIRE(result.value() == 0.0);
    result.set_gradient(1.0);
    result.backward();
    REQUIRE(result.gradient() == 1.0);
    const std::vector<Variable> &parameters = neuron.parameters();
    REQUIRE(parameters.size() == n_in + 1);
    for (std::vector<double>::size_type i = 0; i < n_in; i++)
    {
        REQUIRE(parameters[i].gradient() == input[i]);
    }
    REQUIRE(parameters[n_in].gradient() == 1.0);
}
