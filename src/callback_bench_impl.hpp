#ifndef CALLBACK_BENCH_IMPL_HPP
#define CALLBACK_BENCH_IMPL_HPP

#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>

#ifndef VARMAT
#define CAST_VAR stan::math::promote_scalar<stan::math::var>
#else
#define CAST_VAR make_var_value

template <typename T>
auto make_var_value(const T& x) {
  return stan::math::var_value<T>(x);
}
#endif

template <typename T1, typename T2,
	  stan::require_st_arithmetic<T1>* = nullptr>
auto bench_promote(const T2& x) {
  return x;
}

template <typename T1, typename T2,
	  stan::require_st_var<T1>* = nullptr>
auto bench_promote(const T2& x) {
  return CAST_VAR(x);
}

struct non_vec {
  template <typename F, typename T>
  auto operator()(const F&f, T&& args_tuple) {
    auto no_vec_impl = [&f](const auto& a, const auto&... args) {
      decltype(f(a, args...)) total = 0.0;
      for(size_t i = 0; i < a.size(); ++i) {
	total += f(a.coeff(i), args.coeff(i)...);
      }
      return total;
    };

    return stan::math::apply(no_vec_impl, args_tuple);
  }
};

struct vec {
  template<typename F, typename T>
  auto operator()(const F& f, T&& args_tuple) {
    return stan::math::apply(f, args_tuple);
  }
};

template <typename Vectorizer = vec, typename F_init, typename F_run>
static void callback_bench_impl(F_init init, F_run run, benchmark::State& state) {
  using stan::math::var;

  Vectorizer vectorizer;

  Eigen::MatrixXd x_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  Eigen::MatrixXd y_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));

  for (auto _ : state) {
    auto init_tuple = init(state);

    auto start = std::chrono::high_resolution_clock::now();
    var lp = 0;
    lp += vectorizer(run, init_tuple);
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}

#endif
