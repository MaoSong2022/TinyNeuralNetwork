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

    std::vector<Variable> weights()
    {
        return _weights;
    }

    Variable bias()
    {
        return _bias;
    }

    std::vector<Variable> parameters()
    {
        std::vector<Variable> result;
        result.insert(result.end(), _weights.begin(), _weights.end());
        result.push_back(_bias);
        return result;
    }

    Variable forward(const std::vector<Variable> &inputs);
};
