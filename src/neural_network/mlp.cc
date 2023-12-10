#include "mlp.h"


std::vector<Variable> MLP::parameters() const
{
    std::vector<Variable> result;
    for (const auto &layer : _layers)
    {
        result.insert(result.end(),
                      layer.parameters().begin(),
                      layer.parameters().end());
    }
    return result;
}

std::vector<Variable> MLP::forward(const std::vector<double> &inputs)
{
    std::vector<Variable> result = _layers[0].forward(inputs);
    for (size_t i = 1; i < _layers.size(); i++)
    {
        result = _layers[i].forward(value(result));
    }
    return result;
}
