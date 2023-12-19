#pragma once

#include "../variable/variable.h"


Variable MSELoss(const std::vector<Variable> &predictions,
                 const std::vector<double> &targets);
