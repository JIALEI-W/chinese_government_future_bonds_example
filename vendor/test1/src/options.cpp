#include "options.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string require_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --eps VALUE         target RMSE (default 0.1)\n"
        << "  --N0 INTEGER        pilot paths on each new level (default 100)\n"
        << "  --Lmin INTEGER      initial finest level (default 2)\n"
        << "  --Lmax INTEGER      maximum level (default 20)\n"
        << "  --repeats INTEGER   timing repetitions (default 5)\n"
        << "  --dimension INTEGER model dimension (default 5)\n"
        << "  --seed INTEGER      random-correlation seed\n"
        << "  --output-dir PATH   result directory\n";
}

} // namespace

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--eps") {
            options.eps = std::stod(require_value(argc, argv, index));
        } else if (argument == "--N0") {
            options.N0 = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--Lmin") {
            options.Lmin = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--Lmax") {
            options.Lmax = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--repeats") {
            options.repeats = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--dimension") {
            options.dimension = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                std::stoul(require_value(argc, argv, index)));
        } else if (argument == "--output-dir") {
            options.output_dir = require_value(argc, argv, index);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.eps <= 0.0 || options.N0 < 2 || options.Lmin < 2
        || options.Lmax < options.Lmin || options.Lmax > 20
        || options.repeats < 1 || options.dimension < 2) {
        throw std::invalid_argument("invalid benchmark options");
    }
    return options;
}
