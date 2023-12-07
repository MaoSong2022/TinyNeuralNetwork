
#include "neuron.h"

Variable Neuron::forward(const std::vector<Variable> &inputs)
{
    Variable result = _bias;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        result = result + _weights[i] * inputs[i];
    }
    return result;
}
