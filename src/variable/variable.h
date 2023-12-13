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

    Variable &operator=(Variable &&other)
    {
        this->_value = other._value;
        this->_gradient = other._gradient;
        this->_op = other._op;
        this->_name = other._name;
        this->_children = other._children;
        this->_backward = other._backward;
        other.ref = nullptr;
        this->ref = this;
        return *this;
    }

    Variable(Variable &&other)
    {
        this->_value = other._value;
        this->_gradient = other._gradient;
        this->_op = other._op;
        this->_name = other._name;
        this->_children = other._children;
        this->_backward = other._backward;
        this->ref = other.ref;
        other.ref = nullptr;
        this->ref = this;
    }

    double value() const
    {
        return _value;
    }

    const std::string &name() const
    {
        return _name;
    }

    void set_name(const std::string &name)
    {
        _name = name;
    }

    void set_value(double value)
    {
        _value = value;
    }

    const std::string &op() const
    {
        return _op;
    }

    void set_op(const std::string &op)
    {
        _op = op;
    }

    void set_backward(std::function<void(Variable *)> backward)
    {
        _backward = backward;
    }

    double gradient() const
    {
        return _gradient;
    }

    void set_gradient(double gradient)
    {
        _gradient = gradient;
    }

    Variable *reference() const
    {
        return ref;
    }

    void set_ref(Variable *reference)
    {
        this->ref = reference;
    }

    const std::vector<Variable> &children() const
    {
        return _children;
    }

    std::vector<Variable> &mutable_children()
    {
        return _children;
    }

    void set_children(const std::vector<Variable> &children)
    {
        _children = children;
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

    Variable activate(std::string activate_function);

public:
    friend std::ostream &operator<<(std::ostream &os, const Variable &var);

    Variable operator+(const Variable &other);
    Variable operator-(const Variable &other);
    Variable operator*(const Variable &other);
    Variable operator/(const Variable &other);
    Variable operator-() const;
    Variable identity() const;

    Variable operator+(const double other) const;
    Variable operator-(const double other) const;
    Variable operator*(const double other) const;
    Variable operator/(const double other) const;
    friend Variable operator+(const double other, const Variable &var);
    friend Variable operator-(const double other, const Variable &var);
    friend Variable operator*(const double other, const Variable &var);
    friend Variable operator/(const double other, const Variable &var);
    friend Variable dot_product(const std::vector<Variable> &a,
                                const std::vector<Variable> &b);
    friend Variable dot_product(const std::vector<Variable> &a,
                                const std::vector<double> &b);

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
