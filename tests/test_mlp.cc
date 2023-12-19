#include "loss.h"
#include "mlp.h"
#include <catch2/catch.hpp>

TEST_CASE("Test mlp", "[MLP]")
{
    SECTION("Test one layer")
    {
        MLP mlp(3, std::vector<size_t>{1});
        REQUIRE(mlp.layers().size() == 1);
        REQUIRE(mlp.layers()[0].neurons().size() == 1);
        REQUIRE(mlp.parameters().size() == 4);

        std::vector<double> inputs{2.0, 3.0, -1.0};
        std::vector<double> targets{1.0};

        std::vector<Variable> results = mlp.forward(inputs);
        REQUIRE(results.size() == 1);
        Variable loss = MSELoss(results, targets);
        loss.set_gradient(1.0);
        loss.backward();

        REQUIRE(loss.children().size() == 1);
        const auto &child = loss.children()[0];
        REQUIRE(child.reference() == &results[0]);
        REQUIRE(child.gradient() ==
                Approx(loss.gradient() * 2 * (child.value() - targets[0])));
        REQUIRE(child.children().size() == 1);
        const auto &grandson = child.children()[0];
        REQUIRE(
            grandson.gradient() ==
            Approx(child.gradient() * (1.0 - child.value() * child.value())));
        const auto &layer = mlp.layers()[0];
        const auto &parameters = layer.parameters();
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(parameters[i].gradient() ==
                    Approx(grandson.gradient() * inputs[i]));
        }
        REQUIRE(parameters[3].gradient() == Approx(grandson.gradient()));

        std::vector<double> old_values(4);
        for (size_t i = 0; i < 4; ++i)
        {
            old_values[i] = mlp.parameters()[i].value();
        }

        for (auto &parameter : mlp.parameters())
        {
            parameter.gradient_descent(0.1);
        }

        for (size_t i = 0; i < 4; ++i)
        {
            REQUIRE(
                mlp.parameters()[i].value() ==
                Approx(old_values[i] - 0.1 * mlp.parameters()[i].gradient()));
        }
    }

    SECTION("Test two layer")
    {
        MLP mlp(3, std::vector<size_t>{2, 1});
        REQUIRE(mlp.layers().size() == 2);
        REQUIRE(mlp.layers()[0].neurons().size() == 2);
        REQUIRE(mlp.layers()[1].neurons().size() == 1);
        REQUIRE(mlp.parameters().size() == 11);

        std::vector<double> inputs{2.0, 3.0, -1.0};
        std::vector<double> targets{1.0};

        std::vector<Variable> results = mlp.forward(inputs);
        REQUIRE(results.size() == 1);
        Variable loss = MSELoss(results, targets);
        loss.set_gradient(1.0);
        loss.backward();

        REQUIRE(loss.children().size() == 1);
        const auto &child = loss.children()[0];
        REQUIRE(child.reference() == &results[0]);
        REQUIRE(child.gradient() ==
                Approx(loss.gradient() * 2 * (child.value() - targets[0])));
        REQUIRE(child.children().size() == 1);
        const auto &grandson = child.children()[0];
        REQUIRE(
            grandson.gradient() ==
            Approx(child.gradient() * (1.0 - child.value() * child.value())));
        const auto &layer = mlp.layers()[0];
        const auto &parameters = layer.parameters();
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(parameters[i].gradient() ==
                    Approx(child.gradient() * inputs[i]));
        }
        REQUIRE(parameters[3].gradient() == Approx(child.gradient()));

        std::vector<double> old_values(4);
        for (size_t i = 0; i < 4; ++i)
        {
            old_values[i] = parameters[i].value();
        }

        for (auto &parameter : mlp.parameters())
        {
            parameter.gradient_descent(0.1);
        }

        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(parameters[i].value() ==
                    Approx(old_values[i] - 0.1 * parameters[i].gradient()));
        }
        REQUIRE(parameters[3].value() ==
                Approx(old_values[3] - 0.1 * parameters[3].gradient()));
    }
}
