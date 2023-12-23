# An C++ implemented Neural Network
This is a C++ implemented, a simple implementation of Multi-Layer Perception (MLP), it is enclosed so that we can use it in the same way of using Pytorch.

Here is a simple example:
```
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


## Building

First, clone this repo and do the preliminary work:

```shell
git clone --recursive https://github.com/franneck94/CppProjectTemplate
make prepare
```

- App Executable

```shell
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target main
cd app
./main
```

- Unit testing

```shell
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug --target unit_tests
cd tests
./unit_tests
```

- Documentation

```shell
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug --target docs
```

- Code Coverage (Unix only)

```shell
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
cmake --build . --config Debug --target coverage
```

For more info about CMake see [here](./README_cmake.md).
The first time to compile may take a long time, be patient!

# Method
Since the forward process and backward process are asynchronous, it is necessary for us to remember what happened before, this project uses a special class `Variable`, which has a special member `_backward` of function type to determine the behavior of backward process. In other words, `Variable` class passes gradient related information during backward process.

Then, `Neuron`, `Layer`, `MLP` are built step by step, of which `_weights` and `_bias` are the cores, which we use a special field of `Variable` called `ref` to ensure they are correctly updated.



# Checklists
[ ] support batch inputs.  
[ ] support convolution neural networks.   
[ ] support more loss functions.  
[ ] memory use efficiency.




## Structure
``` text
├── CMakeLists.txt
├── app
│   ├── CMakesLists.txt
│   └── main.cc
├── cmake
│   └── cmake modules
├── docs
│   ├── Doxyfile
│   └── html/
├── external
│   ├── CMakesLists.txt
│   ├── ...
├── src
│   ├── CMakesLists.txt
│   ├── my_lib.h
│   └── my_lib.cc
└── tests
    ├── CMakeLists.txt
    └── main.cc
```

Library code goes into [src/](src/), main program code in [app/](app) and tests go in [tests/](tests/).

## Software Requirements

- CMake 3.21+
- GNU Makefile
- Doxygen
- Conan or VCPKG
- MSVC 2017 (or higher), G++9 (or higher), Clang++9 (or higher)
- Optional: Code Coverage (only on GNU|Clang): lcov, gcovr
- Optional: Makefile, Doxygen, Conan, VCPKG




# Acknowledgement
1. This project is inspired by lectures given by [Andrej Karpathy](https://karpathy.ai/),  here is the [video link](https://www.youtube.com/watch?v=VMj-3S1tku0), and the corresponding [Github Repository](https://github.com/karpathy/micrograd).
2. This project uses the template provided by [CppProjectTemplate](https://github.com/franneck94/CppProjectTemplate).
3. The documentation is generated with the help of [Doxygen](https://www.doxygen.nl/index.html).
