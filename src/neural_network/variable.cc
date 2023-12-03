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
    return -right + left;
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
