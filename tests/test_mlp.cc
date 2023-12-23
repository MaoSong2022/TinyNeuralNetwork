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

        std::vector<Variable> &results = mlp.forward(inputs);
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
            REQUIRE(parameters[i].reference()->gradient() ==
                    Approx(grandson.gradient() * inputs[i]));
        }
        REQUIRE(parameters[3].reference()->gradient() ==
                Approx(grandson.gradient()));

        std::vector<double> old_values(4);
        for (size_t i = 0; i < 4; ++i)
        {
            old_values[i] = mlp.parameters()[i].value();
        }

        for (auto &parameter : mlp.mutable_parameters())
        {
            parameter.gradient_descent(0.1);
        }

        for (size_t i = 0; i < 4; ++i)
        {
            REQUIRE(mlp.parameters()[i].reference()->value() ==
                    Approx(old_values[i] -
                           0.1 * mlp.parameters()[i].reference()->gradient()));
        }

        std::vector<Variable> &new_results = mlp.forward(inputs);
        Variable new_loss = MSELoss(new_results, targets);
        REQUIRE(new_loss.value() <= Approx(loss.value()));
    }

    SECTION("Test two layer")
    {
        MLP mlp(3, std::vector<size_t>{2, 1});
        REQUIRE(mlp.layers().size() == 2);
        REQUIRE(mlp.layers()[0].neurons().size() == 2);
        REQUIRE(mlp.layers()[0].parameters().size() == 8);
        REQUIRE(mlp.layers()[1].neurons().size() == 1);
        REQUIRE(mlp.layers()[1].parameters().size() == 3);
        REQUIRE(mlp.parameters().size() == 11);

        std::vector<double> inputs{2.0, 3.0, -1.0};
        std::vector<double> targets{2.0};

        std::vector<Variable> &results = mlp.forward(inputs);
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
        REQUIRE(grandson.children().size() == 2);
        REQUIRE(
            grandson.gradient() ==
            Approx(child.gradient() * (1.0 - child.value() * child.value())));
        REQUIRE(grandson.children()[0].gradient() ==
                Approx(grandson.gradient()));
        REQUIRE(grandson.children()[1].gradient() ==
                Approx(grandson.gradient()));
        REQUIRE(grandson.children()[1].reference() ==
                mlp.layers()[1].parameters()[2].reference());

        const auto &layer1_product = grandson.children()[0];
        REQUIRE(layer1_product.children().size() == 4);
        for (size_t i = 0; i < 2; ++i)
        {
            REQUIRE(layer1_product.children()[i + 2].reference() ==
                    mlp.results()[0][i].reference());
        }
        for (size_t i = 0; i < 2; ++i)
        {
            REQUIRE(layer1_product.children()[i].gradient() ==
                    Approx(layer1_product.gradient() *
                           layer1_product.children()[i + 2].value()));
            REQUIRE(layer1_product.children()[i + 2].gradient() ==
                    Approx(layer1_product.gradient() *
                           layer1_product.children()[i].value()));
        }

        const auto &layer0_results = mlp.results()[0];
        for (const auto &result : layer0_results)
        {
            REQUIRE(result.reference() == &result);
        }
        const auto &neuron0_result = layer0_results[0];
        const auto &neuron1_result = layer0_results[1];
        REQUIRE(neuron0_result.reference() == &neuron0_result);
        REQUIRE(neuron1_result.reference() == &neuron1_result);
        REQUIRE(neuron0_result.children().size() == 1);
        REQUIRE(neuron1_result.children().size() == 1);
        const auto &layer0_activated = neuron0_result.children()[0];
        REQUIRE(layer0_activated.children().size() == 2);
        REQUIRE(layer0_activated.children()[0].gradient() ==
                Approx(layer0_activated.gradient()));
        REQUIRE(layer0_activated.children()[1].gradient() ==
                Approx(layer0_activated.gradient()));
        const auto &layer0_product = layer0_activated.children()[0];
        REQUIRE(layer0_product.children().size() == 3);
        const auto &neuron0_0 = mlp.layers()[0].neurons()[0];
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(layer0_product.children()[i].reference() ==
                    &neuron0_0.weights()[i]);
        }

        for (auto &parameter : mlp.mutable_parameters())
        {
            parameter.gradient_descent(0.1);
        }
        std::vector<Variable> &new_results = mlp.forward(inputs);
        Variable new_loss = MSELoss(new_results, targets);
        REQUIRE(new_loss.value() <= Approx(loss.value()));
    }
}
