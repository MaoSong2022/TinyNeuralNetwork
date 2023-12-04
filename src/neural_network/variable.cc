#include "variable.h"


#include <stdexcept>

Variable Variable::operator-()
{
    Variable result;
    result._value = -this->_value;
    result._children.insert(this);
    result._backward = [&]() { this->_gradient += -result._gradient; };
    return result;
}

Variable Variable::operator+(Variable &other)
{
    Variable result;
    result._value = this->_value + other._value;
    result._children.insert(this);
    result._children.insert(&other);
    result._backward = [&]() {
        this->_gradient += result._gradient;
        other._gradient += result._gradient;
    };
    return result;
}


Variable Variable::operator-(Variable &other)
{
    Variable result;
    result._value = this->_value - other._value;
    result._children.insert(this);
    result._children.insert(&other);
    result._backward = [&]() {
        this->_gradient += result._gradient;
        other._gradient += -result._gradient;
    };
    return result;
}


Variable Variable::operator*(Variable &other)
{
    Variable result;
    result._value = this->_value * other._value;
    result._children.insert(this);
    result._children.insert(&other);
    result._backward = [&]() {
        this->_gradient += result._gradient * other._value;
        other._gradient += result._gradient * this->_value;
    };
    return result;
}


Variable Variable::operator/(Variable &other)
{
    if (other._value == 0)
    {
        throw std::overflow_error("Divide by zero exception");
    }

    Variable result;
    result._value = this->_value / other._value;
    result._children.insert(this);
    result._children.insert(&other);
    result._backward = [&]() {
        this->_gradient += result._gradient / other._value;
        other._gradient +=
            -result._gradient * this->_value / (other._value * other._value);
    };
    return result;
}

void Variable::topological_sort(std::unordered_set<Variable *> &visited,
                                std::stack<Variable *> &stack)
{
    if (visited.find(this) != visited.end())
    {
        return;
    }
    visited.insert(this);
    for (Variable *child : _children)
    {
        child->topological_sort(visited, stack);
    }
    stack.push(this);
}

void Variable::backward()
{
    std::unordered_set<Variable *> visited;
    std::stack<Variable *> stack;
    topological_sort(visited, stack);
    while (!stack.empty())
    {
        stack.top()->_backward();
        stack.pop();
    }
}

Variable Variable::operator+(double value)
{
    Variable result;
    result._value = this->_value + value;
    result._children.insert(this);
    result._backward = [&]() { this->_gradient += result._gradient; };
    return result;
}


Variable Variable::operator-(double value)
{
    Variable result;
    result._value = this->_value - value;
    result._children.insert(this);
    result._backward = [&]() { this->_gradient += result._gradient; };
    return result;
}


Variable Variable::operator*(double value)
{
    Variable result;
    result._value = this->_value * value;
    result._children.insert(this);
    result._backward = [&, value]() {
        this->_gradient += result._gradient * value;
    };

    return result;
}


Variable Variable::operator/(double value)
{
    if (value == 0)
    {
        throw std::overflow_error("Divide by zero exception");
    }
    Variable result = *this * (1.0 / value);
    return result;
}

Variable operator+(double left, Variable &right)
{
    return right + left;
}

Variable operator-(double left, Variable &right)
{
    Variable result;
    result._value = left - right._value;
    result._children.insert(&right);
    result._backward = [&]() { right._gradient += -result._gradient; };
    return result;
}

Variable operator*(double left, Variable &right)
{
    return right * left;
}
Variable operator/(double left, Variable &right)
{
    if (right._value == 0)
    {
        throw std::overflow_error("Divide by zero exception");
    }
    Variable result;
    result._value = left / right._value;
    result._children.insert(&right);
    result._backward = [&]() {
        right._gradient +=
            -left / (right._value * right._value) * result._gradient;
    };
    return result;
}

Variable Variable::exp()
{
    Variable result;
    result._value = std::exp(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += result._gradient * result._value;
    };
    return result;
}

Variable Variable::log()
{
    if (this->_value == 0)
    {
        throw std::overflow_error("Log of zero exception");
    }

    Variable result;
    result._value = std::log(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += result._gradient / this->_value;
    };
    return result;
}

Variable Variable::sin()
{
    Variable result;
    result._value = std::sin(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += result._gradient * std::cos(this->_value);
    };
    return result;
}
Variable Variable::cos()
{
    Variable result;
    result._value = std::cos(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += -result._gradient * std::sin(this->_value);
    };
    return result;
}

Variable Variable::tan()
{
    if (std::fmod(this->_value - M_PI_2, M_PI) == 0)
    {
        throw std::overflow_error("Tan of pi/2 exception");
    }
    Variable result;
    result._value = std::tan(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += result._gradient /
                           (std::cos(this->_value) * std::cos(this->_value));
    };
    return result;
}

Variable Variable::pow(double power)
{
    if (this->_value == 0 && power < 0)
    {
        throw std::overflow_error("Power of zero exception");
    }
    Variable result;
    result._value = std::pow(this->_value, power);
    result._children.insert(this);
    result._backward = [&, power]() {
        this->_gradient +=
            result._gradient * power * std::pow(this->_value, power - 1);
    };
    return result;
}

Variable Variable::sinh()
{
    Variable result;
    result._value = std::sinh(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += result._gradient * std::cosh(this->_value);
    };
    return result;
}

Variable Variable::cosh()
{
    Variable result;
    result._value = std::cosh(this->_value);
    result._children.insert(this);
    result._backward = [&]() {
        this->_gradient += result._gradient * std::sinh(this->_value);
    };
    return result;
}

