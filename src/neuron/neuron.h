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
    std::vector<Variable> _parameters;

public:
    Neuron(size_t n_in, std::string activate_function = "tanh")
        : _weights(n_in), _activate_function(activate_function)
    {
        _parameters.reserve(n_in + 1);
        unsigned seed = static_cast<unsigned>(
            std::chrono::system_clock::now().time_since_epoch().count());
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        distribution.reset();
        for (size_t i = 0; i < n_in; i++)
        {
            _weights[i] = Variable(distribution(generator), 0.0, "", "weights");
            _parameters.push_back(_weights[i]);
        }
        _bias = Variable(distribution(generator), 0, "", "bias");
        _parameters.push_back(_bias);
    }

    Neuron(const Neuron &other)
        : _weights(other._weights), _bias(other._bias),
          _activate_function(other._activate_function){};

    Neuron &operator=(const Neuron &other)
    {
        _weights = other._weights;
        _bias = other._bias;
        _activate_function = other._activate_function;
        return *this;
    }

    Neuron(Neuron &&other) noexcept
        : _weights(other._weights), _bias(other._bias),
          _activate_function(other._activate_function)
    {
        for (auto &weight : other._weights)
        {
            weight.set_ref(nullptr);
        }
        other._weights.clear();
        other._bias.set_ref(nullptr);
        for (auto &weight : _weights)
        {
            weight.set_ref(&weight);
        }
        _bias.set_ref(&_bias);
    }

    Neuron &operator=(Neuron &&other) noexcept
    {
        _weights = other._weights;
        _bias = other._bias;
        _activate_function = other._activate_function;
        for (auto &weight : other._weights)
        {
            weight.set_ref(nullptr);
        }
        other._weights.clear();
        other._bias.set_ref(nullptr);

        for (auto &weight : _weights)
        {
            weight.set_ref(&weight);
        }
        _bias.set_ref(&_bias);
        return *this;
    }

    ~Neuron()
    {
        for (auto &weight : _weights)
        {
            weight.set_ref(nullptr);
        }
        _bias.set_ref(nullptr);
    };

    const std::vector<Variable> &weights() const
    {
        return _weights;
    }

    const Variable &bias() const
    {
        return _bias;
    }

    const std::vector<Variable> &parameters() const
    {
        return _parameters;
    }

    Variable forward(const std::vector<double> &inputs);
    Variable forward(const std::vector<Variable> &variables);
};
