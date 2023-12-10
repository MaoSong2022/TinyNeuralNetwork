#pragma once

#include <vector>

#include "../layer/layer.h"

class MLP
{
private:
    size_t _n_in;
    std::vector<size_t> _n_outs;
    std::vector<Layer> _layers;

public:
    MLP(size_t n_in, std::vector<size_t> n_outs) : _n_in(n_in), _n_outs(n_outs)
    {
        size_t n_prev = n_in;
        for (size_t n_out : n_outs)
        {
            _layers.push_back(Layer(n_prev, n_out));
            n_prev = n_out;
        }
    }

    std::vector<Variable> parameters() const;

    std::vector<Variable> forward(const std::vector<double> &inputs);
};
