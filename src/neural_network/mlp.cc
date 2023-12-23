#include "mlp.h"


std::vector<Variable> &MLP::forward(const std::vector<double> &inputs)
{
    std::vector<Variable> result = _layers[0].forward(inputs);
    for (size_t i = 1; i < _layers.size(); i++)
    {
        result = _layers[i].forward(result);
    }
    return result;
}
