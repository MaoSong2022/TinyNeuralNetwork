#include <iostream>

#include <nlohmann/json.hpp>

#include "my_lib.h"

int print_hello_world()
{
    std::cout << "Cout: Hello World" << '\n';
    std::cout << NLOHMANN_JSON_VERSION_MAJOR << '\n';

    // Adress Sanitizer should see this :)
    int *x = new int[42];

    return 1;
}
