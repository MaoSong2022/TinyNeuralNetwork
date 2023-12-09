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

std::variant<std::vector<Variable>, Variable> Layer::forward(
    const std::vector<double> &inputs)
{
    // last layer
    if (_n_out == 1)
    {
        return _neurons[0].forward(inputs);
    }

    // intermediate layer
    std::vector<Variable> result;
    for (auto &neuron : _neurons)
    {
        result.push_back(neuron.forward(inputs));
    }
    return result;
}
