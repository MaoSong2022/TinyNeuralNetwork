#include "loss.h"


Variable MSELoss(std::vector<Variable> predictions, std::vector<double> targets)
{
    size_t n = predictions.size();
    Variable sum;
    for (size_t i = 0; i < n; i++)
    {
        sum =
            sum + (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
    }
    return Variable(sum) / static_cast<double>(n);
}
