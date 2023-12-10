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
    Variable a(2.0);
    Variable b(3.0);
    Variable c = a + b;
    std::cout << "JSON: " << NLOHMANN_JSON_VERSION_MAJOR << "."
              << NLOHMANN_JSON_VERSION_MINOR << "."
              << NLOHMANN_JSON_VERSION_PATCH << '\n';
    std::cout << "FMT: " << FMT_VERSION << '\n';
    std::cout << "CXXOPTS: " << CXXOPTS__VERSION_MAJOR << "."
              << CXXOPTS__VERSION_MINOR << "." << CXXOPTS__VERSION_PATCH
              << '\n';
    std::cout << "SPDLOG: " << SPDLOG_VER_MAJOR << "." << SPDLOG_VER_MINOR
              << "." << SPDLOG_VER_PATCH << '\n';
    std::cout << "\n\nUsage Example:\n";

    const auto welcome_message =
        fmt::format("Welcome to {} v{}\n", project_name, project_version);
    spdlog::info(welcome_message);

    cxxopts::Options options(project_name.data(), welcome_message);

    options.add_options("arguments")("h,help", "Print usage")(
        "f,filename",
        "File name",
        cxxopts::value<std::string>())(
        "v,verbose",
        "Verbose output",
        cxxopts::value<bool>()->default_value("false"));

    auto result = options.parse(argc, argv);

    if (argc == 1 || result.count("help"))
    {
        std::cout << options.help() << '\n';
        return 0;
    }

    auto filename = std::string{};
    auto verbose = false;

    if (result.count("filename"))
    {
        filename = result["filename"].as<std::string>();
    }
    else
    {
        return 1;
    }

    verbose = result["verbose"].as<bool>();

    if (verbose)
    {
        fmt::print("Opening file: {}\n", filename);
    }

    auto ifs = std::ifstream{filename};

    if (!ifs.is_open())
    {
        return 1;
    }

    const auto parsed_data = json::parse(ifs);

    if (verbose)
    {
        const auto name = parsed_data["name"];
        fmt::print("Name: {}\n", name);
    }
    std::vector<std::vector<double>> inputs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0},
    };
    std::vector<double> targets = {1.0, -1.0, -1.0, 1.0};

    size_t num_inputs = inputs[0].size();
    std::vector<size_t> num_outputs{4, 4, 1};

    MLP mlp(num_inputs, num_outputs);

    size_t iters = 20;
    double lr = 0.01;

    for (size_t iter = 0; iter < iters; iter++)
    {
        std::vector<Variable> predictions;
        for (const auto &input : inputs)
        {
            std::vector<Variable> result = mlp.forward(input);
            predictions.push_back(result[0]);
        }
        Variable loss = MSELoss(predictions, targets);

        // zero gradient
        for (auto &parameter : mlp.parameters())
        {
            parameter.zero_grad();
        }

        // backward
        loss.backward();

        // update values
        for (auto &parameter : mlp.parameters())
        {
            parameter.gradient_descent(lr);
        }
    }

    return 0;
}
