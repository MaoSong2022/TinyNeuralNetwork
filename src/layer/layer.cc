#include "layer.h"


std::vector<Variable> Layer::parameters() const
{
    std::vector<Variable> result;
    for (const auto &neuron : _neurons)
    {
        result.insert(result.end(),
                      neuron.parameters().begin(),
                      neuron.parameters().end());
    }
    return result;
}

std::vector<Variable> Layer::forward(const std::vector<double> &inputs)
{
    std::vector<Variable> result;
    for (auto &neuron : _neurons)
    {
        result.push_back(neuron.forward(inputs));
    }
    return result;
}


/**
 * Generates a vector of values of the input variables.
 *
 * @param variables the vector of Variable objects to process
 *
 * @return a vector of double values generated from the variables
 */
std::vector<double> value(const std::vector<Variable> &variables)
{
    std::vector<double> result;
    for (const auto &variable : variables)
    {
        result.push_back(variable.value());
    }
    return result;
}
