CXX := clang++
TEST1_ROOT ?= ../vendor/test1
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS := -I$(TEST1_ROOT)/src -include "$(CURDIR)/options_override.hpp"

TEST1_SOURCES := \
	$(TEST1_ROOT)/src/options.cpp \
	$(TEST1_ROOT)/src/path_sampling.cpp \
	$(TEST1_ROOT)/src/statistics.cpp \
	$(TEST1_ROOT)/src/standard_monte_carlo.cpp \
	$(TEST1_ROOT)/src/mlmc_correlation_greeks.cpp \
	$(TEST1_ROOT)/src/validation.cpp \
	$(TEST1_ROOT)/src/benchmark_report.cpp \
	$(TEST1_ROOT)/src/forward_sensitivity.cpp \
	$(TEST1_ROOT)/src/standard_reverse_cholesky.cpp \
	$(TEST1_ROOT)/src/analytic_greeks.cpp \
	$(TEST1_ROOT)/src/cholesky.cpp \
	$(TEST1_ROOT)/src/pathwise_c_adjoint.cpp \
	$(TEST1_ROOT)/src/koerber_cholesky.cpp \
	$(TEST1_ROOT)/src/mlmc_vector.cpp

SOURCES := empirical_main.cpp empirical_benchmark_model.cpp $(TEST1_SOURCES)

.PHONY: all run koerber clean

all: empirical_benchmark koerber_benchmark

plan_standard_only: plan_standard_only.cpp empirical_benchmark_model.cpp $(TEST1_SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

empirical_benchmark: $(SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@

KOERBER_SOURCES := koerber_compare.cpp empirical_benchmark_model.cpp $(TEST1_SOURCES)

koerber_benchmark: $(KOERBER_SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(KOERBER_SOURCES) -o $@

run: empirical_benchmark
	./empirical_benchmark --dimension 4 --eps 0.01 --N0 200 --Lmin 2 --repeats 1 \
		--output-dir ../outputs/test1_empirical_eps001_original

koerber: koerber_benchmark
	./koerber_benchmark --eps 0.01 --N0 200 --Lmin 2 --repeats 5 \
		--output-dir ../outputs/test1_empirical_koerber_eps001_optimized

clean:
	rm -f empirical_benchmark koerber_benchmark
