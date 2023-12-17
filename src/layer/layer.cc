#include "layer.h"


std::vector<Variable> Layer::parameters() const
{
    std::vector<Variable> result(_n_out * (_n_in + 1));
    for (size_t i = 0; i < _n_out; i++)
    {
        const auto &parameters = _neurons[i].parameters();
        for (size_t j = 0; j < _n_in + 1; j++)
        {
            result[i * (_n_in + 1) + j] = parameters[j];
        }
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
