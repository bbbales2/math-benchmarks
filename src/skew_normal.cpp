#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

template <typename T1, typename T2, typename T3, typename T4>
struct init {
  auto operator()(benchmark::State& state) {
  using stan::math::exp;

  Eigen::VectorXd y_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd xi_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd omega_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd alpha_val = Eigen::VectorXd::Random(state.range(0));

  return std::make_tuple(bench_promote<T1>(y_val),
			 bench_promote<T2>(xi_val),
			 bench_promote<T3>(omega_val),
			 bench_promote<T4>(alpha_val));
  }
};

template <typename Vectorizer, typename... Args>
static void skew_normal_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return skew_normal_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void skew_normal_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return skew_normal_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void skew_normal_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return skew_normal_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void skew_normal_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return skew_normal_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(skew_normal_lpdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lpdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_cdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_cdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lcdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lcdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lccdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lccdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(skew_normal_lpdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lpdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_cdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_cdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lcdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lcdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lccdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(skew_normal_lccdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
