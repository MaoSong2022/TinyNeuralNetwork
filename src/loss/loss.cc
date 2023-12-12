#include "loss.h"


Variable MSELoss(std::vector<Variable> predictions, std::vector<double> targets)
{
    size_t n = predictions.size();
    double value = 0;
    for (size_t i = 0; i < n; i++)
    {
        value += (predictions[i].value() - targets[i]) *
                 (predictions[i].value() - targets[i]);
    }
    value /= static_cast<double>(n);

    Variable result;
    result.set_value(value);
    result.set_children(predictions);
    result.set_op("MSELoss");
    result.set_backward([targets](Variable *result) {
        for (size_t i = 0; i < result->children().size(); i++)
        {
            Variable &child = result->mutable_children()[i];
            child.update_gradient(result->gradient() * 2.0 *
                                  (child.value() - targets[i]));
        }
    });

    return result;
}
