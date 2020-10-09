#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include "toss_me.hpp"

static void uniform_custom_nonvectorized(benchmark::State& state) {
  using stan::math::var;

  Eigen::VectorXd alpha_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd beta_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0))) + alpha_val;
  Eigen::VectorXd y_val = (stan::math::inv_logit(Eigen::VectorXd::Random(state.range(0))).array() *
       (beta_val - alpha_val).array()).matrix() + alpha_val;

  for (auto _ : state) {
    auto alpha = stan::math::promote_scalar<stan::math::var>(alpha_val).eval();
    auto beta = stan::math::promote_scalar<stan::math::var>(beta_val).eval();
    auto y = stan::math::promote_scalar<stan::math::var>(y_val).eval();

    auto start = std::chrono::high_resolution_clock::now();
    var lp = 0.0;
    for(size_t i = 0; i < alpha.size(); ++i) {
      lp += uniform_lpdf(y.coeff(i), alpha.coeff(i), beta.coeff(i));
    }
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}

static void uniform_custom_vectorized(benchmark::State& state) {
  using stan::math::var;

  Eigen::VectorXd alpha_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd beta_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0))) + alpha_val;
  Eigen::VectorXd y_val = (stan::math::inv_logit(Eigen::VectorXd::Random(state.range(0))).array() *
       (beta_val - alpha_val).array()).matrix() + alpha_val;

  for (auto _ : state) {
    auto alpha = stan::math::promote_scalar<stan::math::var>(alpha_val).eval();
    auto beta = stan::math::promote_scalar<stan::math::var>(beta_val).eval();
    auto y = stan::math::promote_scalar<stan::math::var>(y_val).eval();

    auto start = std::chrono::high_resolution_clock::now();
    var lp = uniform_lpdf(y, alpha, beta);
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}

int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(uniform_custom_nonvectorized)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(uniform_custom_vectorized)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();

