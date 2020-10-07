#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

auto init(int L, int M) {
  return [L, M](benchmark::State& state) {
    boost::random::mt19937 rng;
    std::vector<std::vector<int>> y;
    Eigen::VectorXd p = stan::math::softmax(Eigen::VectorXd::Random(M));
    for(size_t n = 0; n < state.range(0); ++n) {
      y.push_back(stan::math::multinomial_rng(p, L, rng));
    }

    return std::make_tuple(y,
			   CAST_VAR(p));
  };
};

auto run = [](const auto& ys, const auto& p) {
  stan::math::var total = 0.0;
  for(size_t i = 0; i < ys.size(); ++i) {
    stan::math::multinomial_lpmf(ys[i], p);
  }
  return total;
};

static void multinomial_2_lpmf(benchmark::State& state) {
  callback_bench_impl(init(10, 2), run, state);
}
static void multinomial_3_lpmf(benchmark::State& state) {
  callback_bench_impl(init(10, 3), run, state);
}
static void multinomial_8_lpmf(benchmark::State& state) {
  callback_bench_impl(init(10, 8), run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(multinomial_2_lpmf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(multinomial_3_lpmf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(multinomial_8_lpmf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();

