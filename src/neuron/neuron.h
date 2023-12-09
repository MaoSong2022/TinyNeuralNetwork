#pragma once

#include <vector>

#include "../variable/variable.h"

class Neuron
{
private:
    std::vector<Variable> _weights;
    Variable _bias;
    std::string _activate_function;

public:
    Neuron(size_t n_in, std::string activate_function = "tanh")
        : _weights(std::vector<Variable>(n_in)), _bias(0.0),
          _activate_function(activate_function){};

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
