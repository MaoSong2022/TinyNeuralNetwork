#pragma once

#include <deque>
#include <functional>
#include <unordered_set>


class Variable
{
private:
    double _value;
    double _gradient;
    std::function<void()> _backward;
    std::unordered_set<Variable *> _children;

    void topological_sort(std::unordered_set<Variable *> &visited,
                          std::deque<Variable *> &stack);

public:
    Variable() : _value(0), _gradient(0){};
    Variable(double _value) : _value(_value), _gradient(0){};
    Variable(double _value, double _gradient)
        : _value(_value), _gradient(_gradient){};

    /**
     * Returns the value.
     *
     * @return the value of the variable.
     */
    double value() const
    {
        return _value;
    }
    /**
     * Returns the gradient value.
     *
     * @return the gradient value of the variable.
     */
    double gradient() const
    {
        return _gradient;
    }

    double &mutable_value()
    {
        return _value;
    }
    double &mutable_gradient()
    {
        return _gradient;
    }

    void backward();

    Variable operator+(Variable &other);
    Variable operator-(Variable &other);
    Variable operator*(Variable &other);
    Variable operator/(Variable &other);
};
