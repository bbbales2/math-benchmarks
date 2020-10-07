#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

template <typename T1, typename T2, typename T3>
struct init {
  auto operator()(benchmark::State& state) {
    Eigen::VectorXd y_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0)));
    Eigen::VectorXd y_min_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0))) + y_val;
    Eigen::VectorXd alpha_val = stan::math::exp(Eigen::VectorXd::Random(state.range(0)));

    return std::make_tuple(bench_promote<T1>(y_val),
			   bench_promote<T2>(y_min_val),
			   bench_promote<T3>(alpha_val));
  }
};

template <typename Vectorizer, typename... Args>
static void pareto_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return pareto_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void pareto_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return pareto_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void pareto_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return pareto_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void pareto_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return pareto_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(pareto_lpdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lpdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_cdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_cdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lcdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lcdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lccdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lccdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(pareto_lpdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lpdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_cdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_cdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lcdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lcdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lccdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(pareto_lccdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
