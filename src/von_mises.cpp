#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

template <typename T1, typename T2, typename T3>
struct init {
  auto operator()(benchmark::State& state) {
    using stan::math::exp;
    using stan::math::inv_logit;

      Eigen::VectorXd y_val = Eigen::VectorXd::Random(state.range(0));
      Eigen::VectorXd mu_val = Eigen::VectorXd::Random(state.range(0));
      Eigen::VectorXd kappa_val = exp(Eigen::VectorXd::Random(state.range(0)));

      return std::make_tuple(bench_promote<T1>(y_val),
    			 bench_promote<T2>(mu_val),
    			 bench_promote<T3>(kappa_val));
  }
};

template <typename Vectorizer, typename... Args>
static void von_mises_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return von_mises_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void von_mises_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return von_mises_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void von_mises_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return von_mises_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void von_mises_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return von_mises_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(von_mises_lpdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(von_mises_lpdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(von_mises_lpdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(von_mises_lpdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
