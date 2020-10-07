#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

auto init(int L) {
  return [L](benchmark::State& state) {
    boost::random::mt19937 rng;
    std::vector<int> y;
    std::vector<int> Ls;
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    for(size_t n = 0; n < state.range(0); ++n) {
      y.push_back(stan::math::binomial_rng(L, p, rng));
      Ls.push_back(L);
    }

    return std::make_tuple(y, Ls, CAST_VAR(p));
  };
};

auto run = [](const auto& y, const auto& Ls, const auto& p) {
  return stan::math::binomial_lpmf(y, Ls, p);
};

static void binomial_10_lpmf(benchmark::State& state) {
  callback_bench_impl(init(10), run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(binomial_10_lpmf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();

