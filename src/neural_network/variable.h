#pragma once

#include <functional>
#include <iostream>
#include <stack>
#include <unordered_set>

class Variable
{
private:
    double _value;
    double _gradient;
    std::function<void()> _backward = [] {};
    std::unordered_set<Variable *> _children;

    void topological_sort(std::unordered_set<Variable *> &visited,
                          std::stack<Variable *> &stack);

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

    const std::unordered_set<Variable *> &children() const
    {
        return _children;
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


public:
    Variable operator-();

    Variable operator+(Variable &other);
    Variable operator-(Variable &other);
    Variable operator*(Variable &other);
    Variable operator/(Variable &other);

    Variable operator+(double left);
    Variable operator-(double left);
    Variable operator*(double left);
    Variable operator/(double left);

    friend Variable operator+(double left, Variable &right);
    friend Variable operator-(double left, Variable &right);
    friend Variable operator*(double left, Variable &right);
    friend Variable operator/(double left, Variable &right);
    friend std::ostream &operator<<(std::ostream &os, const Variable &v);

    Variable exp();
    Variable log();
    Variable sin();
    Variable cos();
    Variable tan();
    Variable pow(double power);
    Variable sinh();
    Variable cosh();

    Variable relu();
    Variable tanh();
    Variable sigmoid();
};
