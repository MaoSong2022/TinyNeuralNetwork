#include "variable.h"
#include <fmt/format.h>
#include <math.h>

std::ostream &operator<<(std::ostream &os, const Variable &var)
{
    os << fmt::format("Variable(name: {}, value: {}, gradient: {}, op: {})",
                      var._name,
                      var._value,
                      var._gradient,
                      var._op);
    return os;
}

Variable Variable::operator+(const Variable &other)
{
    Variable result;
    result._value = _value + other._value;
    result._op = "+";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this, other};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(result->_gradient);
        result->_children[1].update_gradient(result->_gradient);
    };
    return result;
}


Variable Variable::operator-(const Variable &other)
{
    Variable result;
    result._value = _value - other._value;
    result._op = "-";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this, other};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(result->_gradient);
        result->_children[1].update_gradient(-result->_gradient);
    };
    return result;
}


Variable Variable::operator*(const Variable &other)
{
    Variable result;
    result._value = _value * other._value;
    result._op = "*";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this, other};
    result._backward = [](Variable *result) {
        Variable &left = result->_children[0];
        Variable &right = result->_children[1];
        left.update_gradient(result->_gradient * right.value());
        right.update_gradient(result->_gradient * left.value());
    };
    return result;
}


Variable Variable::operator/(const Variable &other)
{
    if (other._value == 0)
    {
        throw std::overflow_error("Division by zero");
    }
    Variable result;
    result._value = _value / other._value;
    result._op = "/";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this, other};
    result._backward = [](Variable *result) {
        const Variable &left = result->_children[0];
        const Variable &right = result->_children[1];
        result->_children[0].update_gradient(result->_gradient / right.value());
        result->_children[1].update_gradient(-result->_gradient * left.value() /
                                             (right.value() * right.value()));
    };
    return result;
}

Variable Variable::operator-() const
{
    Variable result;
    result._value = -_value;
    result._op = "-";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(-result->_gradient);
    };
    return result;
}

Variable Variable::identity() const
{
    Variable result;
    result._value = _value;
    result._op = "identity";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(result->_gradient);
    };
    return result;
}


Variable Variable::operator+(const double other) const
{
    Variable result;
    result._value = _value + other;
    result._op = "+";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(result->_gradient);
    };
    return result;
}


Variable Variable::operator-(const double other) const
{
    Variable result;
    result._value = _value - other;
    result._op = "-";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(result->_gradient);
    };
    return result;
}


Variable Variable::operator*(const double other) const
{
    Variable result;
    result._value = _value * other;
    result._op = "*";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [other](Variable *result) {
        result->_children[0].update_gradient(result->_gradient * other);
    };
    return result;
}


Variable Variable::operator/(const double other) const
{
    if (other == 0)
    {
        throw std::overflow_error("Division by zero");
    }
    Variable result;
    result._value = _value / other;
    result._op = "/";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [other](Variable *result) {
        result->_children[0].update_gradient(result->_gradient / other);
    };
    return result;
}


Variable operator+(const double other, const Variable &var)
{
    return var + other;
}


Variable operator-(const double other, const Variable &var)
{
    return -var + other;
}


Variable operator*(const double other, const Variable &var)
{
    return var * other;
}


Variable operator/(const double other, const Variable &var)
{
    return var.pow(-1.0) * other;
}

Variable Variable::pow(const double other) const
{
    if (this->value() == 0 && other < 0)
    {
        throw std::overflow_error("Negative power of zero");
    }
    Variable result;
    result._value = std::pow(_value, other);
    result._op = "pow";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    double value = this->value();
    result._backward = [value, other](Variable *result) {
        result->_children[0].update_gradient(result->_gradient * other *
                                             std::pow(value, other - 1));
    };
    return result;
}

Variable Variable::exp() const
{
    Variable result;
    result._value = std::exp(_value);
    result._op = "exp";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(result->_gradient *
                                             result->_value);
    };
    return result;
}
Variable Variable::log() const
{
    if (_value <= 0)
    {
        throw std::overflow_error("Log of Non-positive number");
    }
    Variable result;
    result._value = std::log(_value);
    result._op = "log";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(result->_gradient / input.value());
    };
    return result;
}
Variable Variable::sin() const
{
    Variable result;
    result._value = std::sin(_value);
    result._op = "sin";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(result->_gradient *
                                             std::cos(input.value()));
    };
    return result;
}
Variable Variable::cos() const
{
    Variable result;
    result._value = std::cos(_value);
    result._op = "cos";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(result->_gradient *
                                             -std::sin(input.value()));
    };
    return result;
}
Variable Variable::tan() const
{
    if (std::fmod(_value - M_PI_2, M_PI) == 0)
    {
        throw std::overflow_error("tan of (2*k*pi+pi)/2");
    }
    Variable result;
    result._value = std::tan(_value);
    result._op = "tan";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(
            result->_gradient /
            (std::cos(input.value()) * std::cos(input.value())));
    };
    return result;
}
Variable Variable::sinh() const
{
    Variable result;
    result._value = std::sinh(_value);
    result._op = "sinh";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(result->_gradient *
                                             std::cosh(input.value()));
    };
    return result;
}
Variable Variable::cosh() const
{
    Variable result;
    result._value = std::cosh(_value);
    result._op = "cosh";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(result->_gradient *
                                             std::sinh(input.value()));
    };
    return result;
}

Variable Variable::tanh() const
{
    Variable result;
    result._value = std::tanh(_value);
    result._op = "tanh";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(
            result->_gradient * (1 - result->_value * result->_value));
    };
    return result;
}
Variable Variable::relu() const
{
    Variable result;
    result._value = _value > 0 ? _value : 0;
    result._op = "relu";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        const Variable &input = result->_children[0];
        result->_children[0].update_gradient(result->_gradient *
                                             (input.value() > 0 ? 1 : 0));
    };
    return result;
}
Variable Variable::sigmoid() const
{
    Variable result;
    result._value = 1 / (1 + std::exp(-_value));
    result._op = "sigmoid";
    result._name =
        fmt::format("Variable({}, {})", result.value(), result.gradient());
    result.ref = nullptr;
    result._children = std::vector<Variable>{*this};
    result._backward = [](Variable *result) {
        result->_children[0].update_gradient(
            result->_gradient * result->_value * (1 - result->_value));
    };
    return result;
}
