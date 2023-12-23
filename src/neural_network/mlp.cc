#include "mlp.h"


std::vector<Variable> &MLP::forward(const std::vector<double> &inputs)
{
    _results[0] = _layers[0].forward(inputs);
    for (size_t i = 1; i < _layers.size(); i++)
    {
        _results[i] = _layers[i].forward(_results[i - 1]);
    }

    return _results.back();
}
