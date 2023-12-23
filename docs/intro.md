# An C++ implemented Neural Network
This is a C++ implemented, a simple implementation of Multi-Layer Perception (MLP), it is enclosed so that we can use it in the same way of using Pytorch.

Here is a simple example:
```c++
int main()
{
    // training data
    std::vector<double> inputs = {2.0, 3.0, -1.0};
    std::vector<double> targets = {1.0};

    // construct mlp
    size_t num_inputs = inputs.size();
    std::vector<size_t> num_outputs{4, 4, 1};
    MLP mlp(num_inputs, num_outputs);

    // configure training parameters
    size_t iters = 20;
    double lr = 0.01;

    for (size_t iter = 0; iter < iters; iter++)
    {
        // forward pass
        const std::vector<Variable> &predictions = mlp.forward(inputs);

        // compute and record loss
        Variable loss = MSELoss(predictions, targets);
        spdlog::info(
            fmt::format("Iteration: {}. Loss: {}", iter, loss.value()));

        // zero gradient
        for (auto &parameter : mlp.mutable_parameters())
        {
            parameter.zero_grad();
        }

        // set gradient in order to back propagate
        loss.set_gradient(1.0);

        // backward
        loss.backward();

        // update values
        for (auto &parameter : mlp.mutable_parameters())
        {
            parameter.gradient_descent(lr);
        }
    }

    return 0;
}
```
for more details, see `app/simple_mlp.cc`.


# Installation
To use the repository, first clone the repository:
```
git clone https://github.com/MaoSong2022/TinyNeuralNetwork.git
```
Then, go to the repository folder:
```
cd TinyNeuralNetwork
```
now, we need to build the targets:
```
make prepare
```

# Example use
To see the output of a simple example, build and run `main`:
```
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target main
cd app
./main
``` 
The first time to compile may take a long time, be patient!

# Method
Since the forward process and backward process are asynchronous, it is necessary for us to remember what happened before, this project uses a special class `Variable`, which has a special member `_backward` of function type to determine the behavior of backward process. In other words, `Variable` class passes gradient related information during backward process.

# Checklists
[ ] support batch inputs.  
[ ] support convolution neural networks.   
[ ] support more loss functions.  
[ ] memory use efficiency.

# Acknowledgement
1. This project is inspired by lectures given by [Andrej Karpathy](https://karpathy.ai/),  here is the [video link](https://www.youtube.com/watch?v=VMj-3S1tku0), and the corresponding [Github Repository](https://github.com/karpathy/micrograd).
2. This project uses the template provided by [CppProjectTemplate](https://github.com/franneck94/CppProjectTemplate).
3. The documentation is generated with the help of [Doxygen](https://www.doxygen.nl/index.html).
