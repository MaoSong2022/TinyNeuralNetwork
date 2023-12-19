#include "loss.h"
#include <catch2/catch.hpp>

TEST_CASE("Test loss", "[Loss]")
{
    SECTION("Test MSE loss")
    {
        std::vector<Variable> predictions(3);
        for (size_t i = 0; i < 3; ++i)
        {
            predictions[i] =
                Variable(1.0 * i, 0.0, "", "input[" + std::to_string(i) + "]");
        }
        std::vector<double> targets{-1.0, -2.0, -3.0};

        Variable loss = MSELoss(predictions, targets);
        REQUIRE(loss.value() == Approx(11.6666666667));
        REQUIRE(loss.children().size() == 3);

        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(loss.children()[i].reference() == &predictions[i]);
        }

        loss.set_gradient(1.0);
        loss.backward();
        for (size_t i = 0; i < 3; ++i)
        {
            REQUIRE(predictions[i].gradient() ==
                    Approx(loss.gradient() * 2.0 *
                           (predictions[i].value() - targets[i])));
        }
    }
}
