#include "benchmark_model.hpp"
#include "standard_monte_carlo.hpp"

#include <cstdint>
#include <iostream>

int main() {
    const BenchmarkModel model = make_benchmark_model(4, 20260820U);
    const std::uint64_t samples = plan_standard_samples(
        model,
        7,
        200,
        0.01,
        kBiasFraction,
        20260820U ^ 0x22222222U
    );
    std::cout << samples << '\n';
    return 0;
}
