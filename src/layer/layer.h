#pragma once

#include <vector>

#include "../neuron/neuron.h"

/**
 * @class Layer
 * This class represents a layer in a neural network, consisting of multiple neurons.
 */
class Layer
{
private:
    size_t _n_in;  // The number of input connections to the layer.
    size_t _n_out; // The number of output connections from the layer.
    std::string _activate_function =
        "tanh";                        // The activation function of the layer.
    std::vector<Neuron> _neurons;      // The neurons in the layer.
    std::vector<Variable> _parameters; // All parameters of the layer.

public:
    /**
     * Constructs a layer with the specified number of input and output connections, and activation function.
     * @param n_in The number of input connections.
     * @param n_out The number of output connections.
     * @param activate_function The activation function of the layer. Default is "tanh".
     */
    Layer(size_t n_in, size_t n_out, std::string activate_function = "tanh")
        : _n_in(n_in), _n_out(n_out), _activate_function(activate_function)
    {
        _parameters.reserve((n_in + 1) * n_out);
        _neurons.reserve(n_out);
        for (size_t i = 0; i < n_out; i++)
        {
            _neurons.emplace_back(n_in, _activate_function);
            for (size_t j = 0; j < _neurons[i].parameters().size(); j++)
            {
                _parameters.push_back(_neurons[i].parameters()[j]);
            }
        }
    };

    /**
     * Copy constructor.
     * @param other The layer to be copied.
     */
    Layer(const Layer &other)
        : _n_in(other._n_in), _n_out(other._n_out),
          _activate_function(other._activate_function),
          _neurons(other._neurons){};

    /**
     * Copy assignment operator.
     * @param other The layer to be assigned.
     * @return A reference to the assigned layer.
     */
    Layer &operator=(const Layer &other)
    {
        _n_in = other._n_in;
        _n_out = other._n_out;
        _activate_function = other._activate_function;
        _neurons = other._neurons;
        return *this;
    }

    /**
     * Move constructor.
     * @param other The layer to be moved.
     */
    Layer &operator=(Layer &&other) noexcept
    {
        _n_in = other._n_in;
        _n_out = other._n_out;
        _activate_function = std::move(other._activate_function);
        _neurons = std::move(other._neurons);
        return *this;
    }

    /**
     * Move assignment operator.
     * @param other The layer to be assigned.
     * @return A reference to the assigned layer.
     */
    Layer(Layer &&other) noexcept
    {
        _n_in = other._n_in;
        _n_out = other._n_out;
        _activate_function = std::move(other._activate_function);
        _neurons = std::move(other._neurons);
    }

    /**
     * Returns all parameters of the layer, including parameters of all neurons.
     * @return The parameters.
     */
    const std::vector<Variable> &parameters() const
    {
        return _parameters;
    }

    /**
     * Returns the neurons in the layer.
     * @return The neurons.
     */
    const std::vector<Neuron> &neurons() const
    {
        return _neurons;
    }

    /**
     * Computes the forward pass of the layer given a vector of input values.
     * @param inputs The input values.
     * @return The output values of the layer as a vector of Variables.
     */
    std::vector<Variable> forward(const std::vector<double> &inputs);

    /**
     * Computes the forward pass of the layer given a vector of input variables.
     * @param variables The input variables.
     * @return The output values of the layer as a vector of Variables.
     */
    std::vector<Variable> forward(const std::vector<Variable> &variables);
};
