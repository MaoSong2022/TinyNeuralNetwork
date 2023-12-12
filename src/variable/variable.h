#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

class Variable
{
private:
    double _value;
    double _gradient;
    std::string _op;
    std::string _name;
    Variable *ref = nullptr;
    std::vector<Variable> _children;
    std::function<void(Variable *)> _backward = [](Variable *) {};

public:
    explicit Variable(double value = 0,
                      double gradient = 0,
                      std::string op = "",
                      std::string name = "")
        : _value(value), _gradient(gradient), _op(op), _name(name), ref(this){};

    Variable(const Variable &other)
    {
        this->_value = other._value;
        this->_gradient = other._gradient;
        this->_op = other._op;
        this->_name = other._name;
        this->_children = other._children;
        this->_backward = other._backward;
        this->ref = other.ref;
    }

    Variable &operator=(const Variable &other)
    {
        this->_value = other._value;
        this->_gradient = other._gradient;
        this->_op = other._op;
        this->_name = other._name;
        this->_children = other._children;
        this->_backward = other._backward;
        this->ref = other.ref;
        return *this;
    }
    double value() const
    {
        return _value;
    }

    void set_value(double value)
    {
        _value = value;
    }

    double gradient() const
    {
        return _gradient;
    }

    void set_gradient(double gradient)
    {
        _gradient = gradient;
    }

    const std::vector<Variable> &children() const
    {
        return _children;
    }

    void backward()
    {
        _backward(this);
        for (auto &child : _children)
        {
            child.backward();
        }
    }

    void update_gradient(double grad)
    {
        _gradient += grad;
        if (this != ref && ref != nullptr)
        {
            ref->update_gradient(grad);
        }
    }

    void zero_grad()
    {
        _gradient = 0;
    }

    void gradient_descent(double lr)
    {
        _value -= lr * _gradient;
    }

public:
    friend std::ostream &operator<<(std::ostream &os, const Variable &var);

    Variable operator+(const Variable &other);
    Variable operator-(const Variable &other);
    Variable operator*(const Variable &other);
    Variable operator/(const Variable &other);
    Variable operator-() const;

    Variable operator+(const double other) const;
    Variable operator-(const double other) const;
    Variable operator*(const double other) const;
    Variable operator/(const double other) const;
    friend Variable operator+(const double other, const Variable &var);
    friend Variable operator-(const double other, const Variable &var);
    friend Variable operator*(const double other, const Variable &var);
    friend Variable operator/(const double other, const Variable &var);

    Variable pow(const double other) const;
    Variable exp() const;
    Variable log() const;
    Variable sin() const;
    Variable cos() const;
    Variable tan() const;
    Variable sinh() const;
    Variable cosh() const;

    Variable tanh() const;
    Variable relu() const;
    Variable sigmoid() const;
};
