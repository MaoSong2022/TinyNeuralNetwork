
#include "neuron.h"

Variable Neuron::forward(const std::vector<double> &inputs)
{
    if (inputs.size() != _weights.size())
    {
        throw std::runtime_error("invalid number of inputs");
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
