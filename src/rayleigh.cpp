#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

template <typename T1, typename T2>
struct init {
  auto operator()(benchmark::State& state) {
    Eigen::VectorXd y_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0)));
    Eigen::VectorXd sigma_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0)));

    return std::make_tuple(bench_promote<T1>(y_val),
			   bench_promote<T2>(sigma_val));
  }
};

template <typename Vectorizer, typename... Args>
static void rayleigh_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return rayleigh_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void rayleigh_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return rayleigh_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void rayleigh_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return rayleigh_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void rayleigh_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return rayleigh_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(rayleigh_lpdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lpdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_cdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_cdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lcdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lcdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lccdf,non_vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lccdf,vec,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(rayleigh_lpdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lpdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_cdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_cdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lcdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lcdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lccdf,non_vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(rayleigh_lccdf,vec,double,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
