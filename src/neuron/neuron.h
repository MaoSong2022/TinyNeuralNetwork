#pragma once

#include <chrono>
#include <random>
#include <vector>

#include "../variable/variable.h"

/**
 * @class Neuron
 * This class represents a neuron in a neural network.
 */
class Neuron
{
private:
    std::vector<Variable> _weights;    // The weights of the neuron.
    Variable _bias;                    // The bias of the neuron.
    std::string _activate_function;    // The activation function of the neuron.
    std::vector<Variable> _parameters; // All parameters of the neuron.

public:
    /**
     * Constructs a neuron with the specified number of input connections and activation function.
     * @param n_in The number of input connections.
     * @param activate_function The activation function of the neuron. Default is "tanh".
     */
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

    /**
     * Copy constructor.
     * @param other The neuron to be copied.
     */
    Neuron(const Neuron &other)
        : _weights(other._weights), _bias(other._bias),
          _activate_function(other._activate_function){};

    /**
     * Copy assignment operator.
     * @param other The neuron to be assigned.
     * @return A reference to the assigned neuron.
     */
    Neuron &operator=(const Neuron &other)
    {
        _weights = other._weights;
        _bias = other._bias;
        _activate_function = other._activate_function;
        return *this;
    }

    /**
     * Move constructor.
     * @param other The neuron to be moved.
     */
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

    /**
     * Move assignment operator.
     * @param other The neuron to be assigned.
     * @return A reference to the assigned neuron.
     */
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

    /**
     * Destructor.
     */
    ~Neuron()
    {
        for (auto &weight : _weights)
        {
            weight.set_ref(nullptr);
        }
        _bias.set_ref(nullptr);
    }

    /**
     * Returns the weights of the neuron.
     * @return The weights.
     */
    const std::vector<Variable> &weights() const
    {
        return _weights;
    }

    /**
     * Returns the bias of the neuron.
     * @return The bias.
     */
    const Variable &bias() const
    {
        return _bias;
    }

    /**
     * Returns all parameters of the neuron, including weights and bias.
     * @return The parameters.
     */
    const std::vector<Variable> &parameters() const
    {
        return _parameters;
    }

    /**
     * Computes the forward pass of the neuron given a vector of input values.
     * @param inputs The input values.
     * @return The output value of the neuron.
     */
    Variable forward(const std::vector<double> &inputs);

    /**
     * Computes the forward pass of the neuron given a vector of input variables.
     * @param variables The input variables.
     * @return The output value of the neuron.
     */
    Variable forward(const std::vector<Variable> &variables);
};
