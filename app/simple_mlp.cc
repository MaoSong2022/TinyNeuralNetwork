// #define FMT_HEADER_ONLY // may need this line

#include <filesystem>
#include <fstream>
#include <iostream>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "config.hpp"
#include "loss.h"
#include "mlp.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

int main()
{
    // training data
    std::vector<double> inputs = {2.0, 3.0, -1.0};
    std::vector<double> targets = {1.0};

    // construct mlp
    size_t num_inputs = inputs.size();
    std::vector<size_t> num_outputs{4, 4, 1};
    MLP mlp(num_inputs, num_outputs);

    // configure training parameters
    size_t iters = 20;
    double lr = 0.01;

    for (size_t iter = 0; iter < iters; iter++)
    {
        // forward pass
        const std::vector<Variable> &predictions = mlp.forward(inputs);

        // compute and record loss
        Variable loss = MSELoss(predictions, targets);
        spdlog::info(
            fmt::format("Iteration: {}. Loss: {}", iter, loss.value()));

        // zero gradient
        for (auto &parameter : mlp.mutable_parameters())
        {
            parameter.zero_grad();
        }

        // set gradient in order to back propagate
        loss.set_gradient(1.0);

        // backward
        loss.backward();

        // update values
        for (auto &parameter : mlp.mutable_parameters())
        {
            parameter.gradient_descent(lr);
        }
    }

    return 0;
}
