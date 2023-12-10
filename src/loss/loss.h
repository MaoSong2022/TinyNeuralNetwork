#pragma once

#include "../variable/variable.h"


Variable MSELoss(std::vector<Variable> predictions,
                 std::vector<double> targets);
