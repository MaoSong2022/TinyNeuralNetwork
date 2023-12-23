#pragma once

#include "../variable/variable.h"


/**
 * Calculates the Mean Squared Error (MSE) loss between a vector of predictions and a vector of targets.
 * @param predictions The vector of predicted values.
 * @param targets The vector of target values.
 * @return The calculated MSE loss as a Variable.
 */
Variable MSELoss(const std::vector<Variable> &predictions,
                 const std::vector<double> &targets);
