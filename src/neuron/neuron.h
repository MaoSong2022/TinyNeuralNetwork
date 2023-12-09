#pragma once

#include <vector>

#include "../neural_network/variable.h"

class Neuron
{
private:
    std::vector<Variable> _weights;
    Variable _bias;


public:
    Neuron(size_t n_in) : _weights(std::vector<Variable>(n_in)), _bias(0.0){};

    const std::vector<Variable> &weights() const
    {
        return _weights;
    }

    const Variable &bias() const
    {
        return _bias;
    }

    std::vector<Variable> parameters() const
    {
        std::vector<Variable> result = _weights;
        result.push_back(_bias);
        return result;
    }

    Variable forward(const std::vector<double> &inputs);
};
