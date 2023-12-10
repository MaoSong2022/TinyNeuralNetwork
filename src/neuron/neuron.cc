
#include "neuron.h"

Variable Neuron::forward(const std::vector<double> &inputs)
{
    if (inputs.size() != _weights.size())
    {
        throw std::runtime_error("invalid number of inputs");
    }
    Variable result = _bias;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        result = result + _weights[i] * inputs[i];
    }

    if (_activate_function == "relu")
    {
        return result.relu();
    }
    if (_activate_function == "sigmoid")
    {
        return result.sigmoid();
    }
    if (_activate_function == "tanh")
    {
        return result.tanh();
    }
    throw std::runtime_error("unknown activation function");
}
