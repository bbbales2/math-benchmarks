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

      Eigen::VectorXd alpha_val = Eigen::VectorXd::Random(state.range(0));
      Eigen::VectorXd beta_val = exp(Eigen::VectorXd::Random(state.range(0))) + alpha_val;
      Eigen::VectorXd y_val = (inv_logit(Eigen::VectorXd::Random(state.range(0))).array() *
       (beta_val - alpha_val).array()).matrix() + alpha_val;

      return std::make_tuple(bench_promote<T1>(y_val),
    			 bench_promote<T2>(alpha_val),
    			 bench_promote<T3>(beta_val));
  }
};

template <typename Vectorizer, typename... Args>
static void uniform_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return uniform_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void uniform_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return uniform_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void uniform_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return uniform_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void uniform_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return uniform_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(uniform_lpdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lpdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_cdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_cdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lcdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lcdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lccdf,non_vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lccdf,vec,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(uniform_lpdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lpdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_cdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_cdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lcdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lcdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lccdf,non_vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(uniform_lccdf,vec,double,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
