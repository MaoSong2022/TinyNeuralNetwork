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


std::vector<Variable> Layer::forward(const std::vector<Variable> &variables)
{
    std::vector<Variable> result;
    for (auto &neuron : _neurons)
    {
        result.push_back(neuron.forward(variables));
    }
    return result;
}
