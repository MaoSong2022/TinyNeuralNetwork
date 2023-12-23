#pragma once

#include <vector>

#include "../layer/layer.h"

/**
 * @class MLP
 * This class represents a Multi-Layer Perceptron (MLP) neural network.
 */
class MLP
{
private:
    size_t _n_in; // The number of input connections to the MLP.
    std::vector<size_t>
        _n_outs; // The number of output connections for each layer in the MLP.
    std::vector<Layer> _layers; // The layers in the MLP.
    std::vector<std::vector<Variable>>
        _results; // The output results for each layer in the MLP.
    std::vector<Variable> _parameters; // All parameters of the MLP.

public:
    /**
     * Constructs an MLP with the specified number of input connections and output connections for each layer.
     * @param n_in The number of input connections.
     * @param n_outs The number of output connections for each layer.
     */
    MLP(size_t n_in, std::vector<size_t> n_outs) : _n_in(n_in), _n_outs(n_outs)
    {
        _layers.reserve(n_outs.size());
        _results.resize(n_outs.size());
        size_t n_prev = n_in;
        size_t num_parameters = 0;
        for (size_t n_out : n_outs)
        {
            num_parameters += n_out * (n_prev + 1);
        }
        _parameters.reserve(num_parameters);

        for (size_t i = 0; i < n_outs.size(); i++)
        {
            _layers.emplace_back(n_prev, n_outs[i], "tanh");
            n_prev = n_outs[i];
            for (size_t j = 0; j < _layers[i].parameters().size(); j++)
            {
                _parameters.push_back(_layers[i].parameters()[j]);
            }
        }
    }

    /**
     * Returns the layers in the MLP.
     * @return The layers.
     */
    const std::vector<Layer> &layers() const
    {
        return _layers;
    }

    /**
     * Returns the output results for each layer in the MLP.
     * @return The output results.
     */
    const std::vector<std::vector<Variable>> &results() const
    {
        return _results;
    }

    /**
     * Returns all parameters of the MLP, including parameters of all layers.
     * @return The parameters.
     */
    const std::vector<Variable> &parameters() const
    {
        return _parameters;
    }

    /**
     * Returns a mutable reference to the parameters of the MLP.
     * @return The mutable parameters.
     */
    std::vector<Variable> &mutable_parameters()
    {
        return _parameters;
    }

    /**
     * Computes the forward pass of the MLP given a vector of input values.
     * @param inputs The input values.
     * @return The output values of the MLP as a vector of Variables.
     */
    std::vector<Variable> &forward(const std::vector<double> &inputs);
};
