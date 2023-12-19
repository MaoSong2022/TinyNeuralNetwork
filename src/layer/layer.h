#pragma once

#include <vector>

#include "../neuron/neuron.h"

class Layer
{
private:
    size_t _n_in;
    size_t _n_out;
    std::string _activate_function = "tanh";
    std::vector<Neuron> _neurons;

public:
    Layer(size_t n_in, size_t n_out, std::string activate_function = "tanh")
        : _n_in(n_in), _n_out(n_out), _activate_function(activate_function)
    {
        for (size_t i = 0; i < n_out; i++)
        {
            _neurons.push_back(Neuron(n_in, activate_function));
        }
    };

    Layer(const Layer &other)
        : _n_in(other._n_in), _n_out(other._n_out),
          _activate_function(other._activate_function),
          _neurons(other._neurons){};

    Layer &operator=(const Layer &other)
    {
        _n_in = other._n_in;
        _n_out = other._n_out;
        _activate_function = other._activate_function;
        _neurons = other._neurons;
        return *this;
    }

    Layer &operator=(Layer &&other) noexcept
    {
        _n_in = other._n_in;
        _n_out = other._n_out;
        _activate_function = std::move(other._activate_function);
        _neurons = std::move(other._neurons);
        return *this;
    }

    Layer(Layer &&other) noexcept
    {
        _n_in = other._n_in;
        _n_out = other._n_out;
        _activate_function = std::move(other._activate_function);
        _neurons = std::move(other._neurons);
    }

    std::vector<Variable> parameters() const;

    const std::vector<Neuron> &neurons() const
    {
        return _neurons;
    }

    std::vector<Variable> forward(const std::vector<double> &inputs);
    std::vector<Variable> forward(const std::vector<Variable> &variables);
};
