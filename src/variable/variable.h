#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

/**
 * @class Variable
 * This class represents a variable in a mathematical expression.
 */
class Variable
{
private:
    double _value;           // The value of the variable.
    double _gradient;        // The gradient of the variable.
    std::string _op;         // The operation associated with the variable.
    std::string _name;       // The name of the variable.
    Variable *ref = nullptr; // A reference to the real variable.
    std::vector<Variable> _children; // The components this variable.
    std::function<void(Variable *)> _backward = [](Variable *) {
    }; //The backward function associated with the variable.

public:
    /**
     * Constructs a new Variable object.
     * @param value The initial value of the variable.
     * @param gradient The initial gradient of the variable.
     * @param op The operation associated with the variable.
     * @param name The name of the variable.
     */
    explicit Variable(double value = 0,
                      double gradient = 0,
                      std::string op = "",
                      std::string name = "")
        : _value(value), _gradient(gradient), _op(op), _name(name), ref(this){};

    /**
     * Copy constructor.
     * @param other The Variable object to copy from.
     */
    Variable(const Variable &other)
        : _value(other._value), _gradient(other._gradient), _op(other._op),
          _name(other._name), ref(other.ref), _children(other._children),
          _backward(other._backward){};

    /**
     * Assignment operator.
     * @param other The Variable object to assign from.
     * @return Reference to the assigned Variable object.
     * @note The reference will be kept.
     */
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

    /**
     * Move assignment operator.
     * @param other The Variable object to move from.
     * @return Reference to the moved Variable object.
     * @note The reference will be kept.
     */
    Variable &operator=(Variable &&other) noexcept
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

    /**
     * Move constructor.
     * @param other The Variable object to move from.
     * @note The reference will be set to this.
     */
    Variable(Variable &&other) noexcept
        : _value(other._value), _gradient(other._gradient), _op(other._op),
          _name(other._name), ref(other.ref), _children(other._children),
          _backward(other._backward)
    {
        other.ref = nullptr;
        this->ref = this;
    };

    /**
     * Destructor.
     */
    ~Variable()
    {
        set_ref(nullptr);
    };

    /**
     * Gets the value of the variable.
     * @return The value of the variable.
     */
    double value() const
    {
        return _value;
    }

    /**
     * Gets the name of the variable.
     * @return The name of the variable.
     */
    const std::string &name() const
    {
        return _name;
    }

    /**
     * Sets the name of the variable.
     * @param name The name of the variable.
     */
    void set_name(const std::string &name)
    {
        _name = name;
    }

    /**
     * Sets the value of the variable.
     * @param value The value of the variable.
     */
    void set_value(double value)
    {
        _value = value;
    }

    /**
     * Gets the operation associated with the variable.
     * @return The operation associated with the variable.
     */
    const std::string &op() const
    {
        return _op;
    }

    /**
     * Sets the operation associated with the variable.
     * @param op The operation associated with the variable.
     */
    void set_op(const std::string &op)
    {
        _op = op;
    }

    /**
     * Sets the backward function associated with the variable.
     * @param backward The backward function associated with the variable.
     */
    void set_backward(std::function<void(Variable *)> backward)
    {
        _backward = backward;
    }

    /**
     * Gets the gradient of the variable.
     * @return The gradient of the variable.
     */
    double gradient() const
    {
        return _gradient;
    }

    /**
     * Sets the gradient of the variable.
     * @param gradient The gradient of the variable.
     */
    void set_gradient(double gradient)
    {
        _gradient = gradient;
    }

    /**
     * Gets the reference to another variable.
     * @return The reference to another variable.
     */
    Variable *reference() const
    {
        return ref;
    }

    /**
     * Sets the reference to another variable.
     * @param reference The reference to another variable.
     */
    void set_ref(Variable *reference)
    {
        this->ref = reference;
    }

    /**
     * Gets the child variables of this variable.
     * @return The child variables of this variable.
     */
    const std::vector<Variable> &children() const
    {
        return _children;
    }

    /**
     * Gets the mutable child variables of this variable.
     * @return The mutable child variables of this variable.
     */
    std::vector<Variable> &mutable_children()
    {
        return _children;
    }

    /**
     * Sets the child variables of this variable.
     * @param children The child variables of this variable.
     */
    void set_children(const std::vector<Variable> &children)
    {
        _children = children;
    }

    /**
     * Performs the backward pass for the variable and its children.
     * @note The backward function is called recursively for each child.
     */
    void backward()
    {
        _backward(this);
        for (auto &child : _children)
        {
            child.backward();
        }
    }

    /**
     * Updates the gradient of the variable.
     * @param grad The gradient update value.
     * @note The gradient is synchronized with the reference variable.
     */
    void update_gradient(double grad)
    {
        _gradient += grad;
        if (this != ref && ref != nullptr)
        {
            ref->update_gradient(grad);
        }
    }

    /**
     * Resets the gradient of the variable to zero.
     * @note The gradient is synchronized with the reference variable.
     */
    void zero_grad()
    {
        _gradient = 0;
        if (this != ref && ref != nullptr)
        {
            ref->zero_grad();
        }
    }

    /**
     * Performs gradient descent optimization on the variable.
     * @param lr The learning rate.
     * @note The gradient is synchronized with the reference variable.
     */
    void gradient_descent(double lr)
    {
        _value -= lr * _gradient;
        if (this != ref && ref != nullptr)
        {
            ref->gradient_descent(lr);
        }
    }

    /**
     * Activate the variable with the given activate function.
     * @param activate_function The name of given activate function.
     * @return the activated variable.
     */
    Variable activate(std::string activate_function);

public:
    /**
     * Overloads the stream insertion operator to print the value of the variable.
     * @param os The output stream.
     * @param var The variable to be printed.
     * @return The modified output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Variable &var);

    /**
     * Overloads the addition operator to perform element-wise addition of two variables.
     * @param other The variable to be added.
     * @return The result of the addition.
     */
    Variable operator+(const Variable &other);

    /**
     * Overloads the subtraction operator to perform element-wise subtraction of two variables.
     * @param other The variable to be subtracted.
     * @return The result of the subtraction.
     */
    Variable operator-(const Variable &other);

    /**
     * Overloads the multiplication operator to perform element-wise multiplication of two variables.
     * @param other The variable to be multiplied.
     * @return The result of the multiplication.
     */
    Variable operator*(const Variable &other);

    /**
     * Overloads the division operator to perform element-wise division of two variables.
     * @param other The variable to be divided.
     * @return The result of the division.
     */
    Variable operator/(const Variable &other);

    /**
     * Overloads the unary negation operator to negate the value of the variable.
     * @return The negated variable.
     */
    Variable operator-() const;

    /**
     * Returns a copy of the variable.
     * @return The copy of the variable.
     */
    Variable identity() const;

    /**
     * Overloads the addition operator to perform element-wise addition of a variable and a scalar.
     * @param other The scalar to be added.
     * @return The result of the addition.
     */
    Variable operator+(const double other) const;

    /**
     * Overloads the subtraction operator to perform element-wise subtraction of a variable and a scalar.
     * @param other The scalar to be subtracted.
     * @return The result of the subtraction.
     */
    Variable operator-(const double other) const;

    /**
     * Overloads the multiplication operator to perform element-wise multiplication of a variable and a scalar.
     * @param other The scalar to be multiplied.
     * @return The result of the multiplication.
     */
    Variable operator*(const double other) const;

    /**
     * Overloads the division operator to perform element-wise division of a variable and a scalar.
     * @param other The scalar to be divided.
     * @return The result of the division.
     */
    Variable operator/(const double other) const;

    /**
     * Overloads the addition operator to perform element-wise addition of a scalar and a variable.
     * @param other The scalar to be added.
     * @param var The variable to be added.
     * @return The result of the addition.
     */
    friend Variable operator+(const double other, const Variable &var);

    /**
     * Overloads the subtraction operator to perform element-wise subtraction of a scalar and a variable.
     * @param other The scalar to be subtracted.
     * @param var The variable to be subtracted.
     * @return The result of the subtraction.
     */
    friend Variable operator-(const double other, const Variable &var);

    /**
     * Overloads the multiplication operator to perform element-wise multiplication of a scalar and a variable.
     * @param other The scalar to be multiplied.
     * @param var The variable to be multiplied.
     * @return The result of the multiplication.
     */
    friend Variable operator*(const double other, const Variable &var);

    /**
     * Overloads the division operator to perform element-wise division of a scalar and a variable.
     * @param other The scalar to be divided.
     * @param var The variable to be divided.
     * @return The result of the division.
     */
    friend Variable operator/(const double other, const Variable &var);

    /**
     * Calculates the dot product of two vectors of variables.
     * @param a The first vector of variables.
     * @param b The second vector of variables.
     * @return The dot product of the two vectors.
     */
    friend Variable dot_product(const std::vector<Variable> &a,
                                const std::vector<Variable> &b);

    /**
     * Calculates the dot product of a vector of variables and a vector of scalars.
     * @param a The vector of variables.
     * @param b The vector of scalars.
     * @return The dot product of the two vectors.
     */
    friend Variable dot_product(const std::vector<Variable> &a,
                                const std::vector<double> &b);

    /**
     * Calculates the power of the variable raised to a scalar exponent.
     * @param other The scalar exponent.
     * @return The result of raising the variable to the exponent.
     */
    Variable pow(const double other) const;

    /**
     * Calculates the exponential of the variable.
     * @return The exponential of the variable.
     */
    Variable exp() const;

    /**
     * Calculates the natural logarithm of the variable.
     * @return The natural logarithm of the variable.
     */
    Variable log() const;

    /**
     * Calculates the sine of the variable.
     * @return The sine of the variable.
     */
    Variable sin() const;

    /**
     * Calculates the cosine of the variable.
     * @return The cosine of the variable.
     */
    Variable cos() const;

    /**
     * Calculates the tangent of the variable.
     * @return The tangent of the variable.
     */
    Variable tan() const;

    /**
     * Calculates the hyperbolic sine of the variable.
     * @return The hyperbolic sine of the variable.
     */
    Variable sinh() const;

    /**
     * Calculates the hyperbolic cosine of the variable.
     * @return The hyperbolic cosine of the variable.
     */
    Variable cosh() const;

    /**
     * Calculates the hyperbolic tangent of the variable.
     * @return The hyperbolic tangent of the variable.
     */
    Variable tanh() const;

    /**
     * Calculates the rectified linear unit (ReLU) of the variable.
     * @return The ReLU of the variable.
     */
    Variable relu() const;

    /**
     * Calculates the sigmoid activation function of the variable.
     * @return The sigmoid of the variable.
     */
    Variable sigmoid() const;
};
