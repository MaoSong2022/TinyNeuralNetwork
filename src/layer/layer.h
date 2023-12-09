#pragma once

#include <vector>

#include "../neuron/neuron.h"

class Layer
{
private:
    size_t _n_in;
    size_t _n_out;
    std::vector<Neuron> _neurons;

public:
    Layer(size_t n_in, size_t n_out)
        : _n_in(n_in), _n_out(n_out),
          _neurons(std::vector<Neuron>(n_out, Neuron(n_in))){};

    std::vector<Variable> parameters() const;

    std::vector<Variable> forward(const std::vector<double> &inputs);
};
