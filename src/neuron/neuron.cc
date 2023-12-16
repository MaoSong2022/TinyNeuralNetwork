
#include "neuron.h"


std::vector<Variable> Neuron::parameters() const
{
    std::vector<Variable> result(_weights.size() + 1);
    for (size_t i = 0; i < _weights.size(); i++)
    {
        result[i] = _weights[i];
    }
    result.back() = _bias;
    return result;
}

Variable Neuron::forward(const std::vector<double> &inputs)
{
    if (inputs.size() != _weights.size())
    {
        throw std::runtime_error("invalid number of inputs");
    }
    Variable result = dot_product(_weights, inputs) + _bias;
    result.set_ref(nullptr);

    return result.activate(_activate_function);
}


Variable Neuron::forward(const std::vector<Variable> &variables)
{
    if (variables.size() != _weights.size())
    {
        throw std::runtime_error("invalid number of inputs");
    }

    Variable result = dot_product(_weights, variables) + _bias;
    result.set_ref(nullptr);

    return result.activate(_activate_function);
}
