#pragma once

#include <chrono>
#include <random>
#include <vector>

#include "../variable/variable.h"

class Neuron
{
private:
    std::vector<Variable> _weights;
    Variable _bias;
    std::string _activate_function;

public:
    Neuron(size_t n_in, std::string activate_function = "tanh")
        : _weights(n_in), _activate_function(activate_function)
    {
        unsigned seed = static_cast<unsigned>(
            std::chrono::system_clock::now().time_since_epoch().count());
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        distribution.reset();
        for (size_t i = 0; i < n_in; i++)
        {
            // _weights[i] = Variable(1.0 * i, 0.0, "", "weights");
            _weights[i] = Variable(distribution(generator), 0.0, "", "weights");
            _weights[i].set_ref(&_weights[i]);
        }
        // _bias = Variable(1.0, 0.0, "", "bias");
        _bias = Variable(distribution(generator), 0, "", "bias");
        _bias.set_ref(&_bias);
    }

    const std::vector<Variable> &weights() const
    {
        return _weights;
    }

    const Variable &bias() const
    {
        return _bias;
    }

    std::vector<Variable> parameters() const;

    Variable forward(const std::vector<double> &inputs);
    Variable forward(const std::vector<Variable> &variables);
};
