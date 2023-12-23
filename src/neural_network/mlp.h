#pragma once

#include <vector>

#include "../layer/layer.h"

class MLP
{
private:
    size_t _n_in;
    std::vector<size_t> _n_outs;
    std::vector<Layer> _layers;
    std::vector<std::vector<Variable>> _results;
    std::vector<Variable> _parameters;

public:
    MLP(size_t n_in, std::vector<size_t> n_outs) : _n_in(n_in), _n_outs(n_outs)
    {
        _layers.reserve(n_outs.size());
        _results.resize(n_outs.size());
        size_t n_prev = n_in;
        size_t num_parameters = 0;
        for (size_t n_out : n_outs)
        {
            num_parameters += n_out * (n_prev + 1);
        }
        _parameters.reserve(num_parameters);

        for (size_t i = 0; i < n_outs.size(); i++)
        {
            _layers.emplace_back(n_prev, n_outs[i], "tanh");
            n_prev = n_outs[i];
            for (size_t j = 0; j < _layers[i].parameters().size(); j++)
            {
                _parameters.push_back(_layers[i].parameters()[j]);
            }
        }
    }

    const std::vector<Layer> &layers() const
    {
        return _layers;
    }

    const std::vector<std::vector<Variable>> &results() const
    {
        return _results;
    }

    const std::vector<Variable> &parameters() const
    {
        return _parameters;
    }

    std::vector<Variable> &mutable_parameters()
    {
        return _parameters;
    }

    std::vector<Variable> &forward(const std::vector<double> &inputs);
};
