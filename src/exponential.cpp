#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

template <typename T1, typename T2>
struct init {
  auto operator()(benchmark::State& state) {
    using stan::math::exp;

    Eigen::VectorXd y_val = exp(Eigen::VectorXd::Random(state.range(0)));
    Eigen::VectorXd beta_val = exp(Eigen::VectorXd::Random(state.range(0)));

    return std::make_tuple(bench_promote<T1>(y_val),
    			 bench_promote<T2>(beta_val));
  }
};

template <typename Vectorizer, typename... Args>
static void exponential_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return exponential_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void exponential_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return exponential_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void exponential_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return exponential_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void exponential_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return exponential_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(exponential_lpdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lpdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_cdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_cdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lcdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lcdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lccdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lccdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(exponential_lpdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lpdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_cdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_cdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lcdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lcdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lccdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(exponential_lccdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
