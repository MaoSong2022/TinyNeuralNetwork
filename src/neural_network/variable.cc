#include "variable.h"


#include <stdexcept>

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
                                std::deque<Variable *> &stack)
{
    if (visited.find(this) != visited.end())
    {
        return;
    }
    visited.insert(this);
    for (auto &child : _children)
    {
        child->topological_sort(visited, stack);
    }
    stack.push_front(this);
}

void Variable::backward()
{
    std::unordered_set<Variable *> visited;
    std::deque<Variable *> stack;
    topological_sort(visited, stack);
    for (auto iter = stack.begin(); iter != stack.end(); ++iter)
    {
        (*iter)->_backward();
    }
}
