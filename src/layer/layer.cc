#include "layer.h"


std::vector<Variable> Layer::forward(const std::vector<double> &inputs)
{
    std::vector<Variable> result(_n_out);
    for (size_t i = 0; i < _n_out; ++i)
    {
        result[i] = _neurons[i].forward(inputs);
    }
    return result;
}


std::vector<Variable> Layer::forward(const std::vector<Variable> &variables)
{
    std::vector<Variable> result(_n_out);
    for (size_t i = 0; i < _n_out; ++i)
    {
        result[i] = _neurons[i].forward(variables);
    }
    return result;
}
