#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct init {
  auto operator()(benchmark::State& state) {
  using stan::math::exp;
    using stan::math::inv_logit;

  Eigen::VectorXd alpha_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd tau_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd beta_val = inv_logit(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd y_val = exp(Eigen::VectorXd::Random(state.range(0))) + tau_val;
  Eigen::VectorXd delta_val = Eigen::VectorXd::Random(state.range(0));

  return std::make_tuple(bench_promote<T1>(y_val),
			 bench_promote<T2>(alpha_val),
			 bench_promote<T3>(tau_val),
			 bench_promote<T4>(beta_val),
			 bench_promote<T5>(delta_val));
  }
};

template <typename Vectorizer, typename... Args>
static void wiener_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return wiener_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(wiener_lpdf,non_vec,var,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(wiener_lpdf,vec,var,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(wiener_lpdf,non_vec,double,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(wiener_lpdf,vec,double,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
