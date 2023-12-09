#pragma once

#include <vector>

#include "../neuron/neuron.h"

class Layer
{
private:
    size_t _n_in;
    size_t _n_out;
    std::string _activate_function = "tanh";
    std::vector<Neuron> _neurons;

public:
    Layer(size_t n_in, size_t n_out, std::string activate_function = "tanh")
        : _n_in(n_in), _n_out(n_out), _activate_function(activate_function),
          _neurons(
              std::vector<Neuron>(n_out, Neuron(n_in, activate_function))){};

    std::vector<Variable> parameters() const;

    std::vector<Variable> forward(const std::vector<double> &inputs);
};
