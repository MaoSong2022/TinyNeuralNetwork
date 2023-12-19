#include "mlp.h"


std::vector<Variable> MLP::parameters() const
{
    std::vector<Variable> result;
    for (size_t i = 0; i < _layers.size(); i++)
    {
        for (size_t j = 0; j < _layers[i].parameters().size(); j++)
        {
            result.push_back(_layers[i].parameters()[j]);
        }
    }
    return result;
}

std::vector<Variable> MLP::forward(const std::vector<double> &inputs)
{
    std::vector<Variable> result = _layers[0].forward(inputs);
    for (size_t i = 1; i < _layers.size(); i++)
    {
        result = _layers[i].forward(result);
    }
    return result;
}
