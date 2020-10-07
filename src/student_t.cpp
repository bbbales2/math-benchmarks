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
  Eigen::VectorXd nu_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd mu_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd sigma_val = exp(Eigen::VectorXd::Random(state.range(0)));

  return std::make_tuple(bench_promote<T1>(y_val),
      bench_promote<T4>(nu_val),
			 bench_promote<T2>(mu_val),
			 bench_promote<T3>(sigma_val));
  }
};

template <typename Vectorizer, typename... Args>
static void student_t_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return student_t_lpdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void student_t_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return student_t_cdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void student_t_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return student_t_lcdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

template <typename Vectorizer, typename... Args>
static void student_t_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return student_t_lccdf(args...);
  };

  callback_bench_impl<Vectorizer>(init<Args...>(), run, state);
}

using stan::math::var;
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK_TEMPLATE(student_t_lpdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lpdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_cdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_cdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lcdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lcdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lccdf,non_vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lccdf,vec,var,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_TEMPLATE(student_t_lpdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lpdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_cdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_cdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lcdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lcdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lccdf,non_vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(student_t_lccdf,vec,double,var,var,var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
