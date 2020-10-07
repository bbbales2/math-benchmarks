MATH ?=math/

-include $(HOME)/.config/stan/make.local
-include make/local
include math/make/compiler_flags
include math/make/dependencies
include math/make/libraries

CXXFLAGS += -I $(BENCHMARK)/include
ifdef VARMAT
CXXFLAGS += -DVARMAT
endif
LDFLAGS += -L $(BENCHMARK)/build/src
LDLIBS += -lbenchmark -ltbb

BENCHMARKS = trace_gen_inv_quad_form_ldlt trace_inv_quad_form_ldlt mdivide_left_ldlt log_determinant_ldlt inv_chi_square gumbel gamma cholesky_decompose eigenvalues_sym eigenvectors_sym sd squared_distance variance sum unit_vector_constrain quad_form trace_quad_form trace_gen_quad_form multiply mdivide_left mdivide_left_spd mdivide_left_tri matrix_power softmax log_softmax log_determinant_spd log_determinant log_sum_exp dot_product divide gp_exp_quad_cov gp_periodic_cov dot_self columns_dot_self tcrossprod determinant inverse multiply_lower_tri_self_transpose columns_dot_product rows_dot_product cauchy chi_square double_exponential exp_mod_normal exponential frechet ordered_constrain positive_ordered_constrain simplex_constrain normal skew_normal student_t logistic lognormal scaled_inv_chi_square inv_gamma weibull rayleigh pareto pareto_type_2 wiener beta beta_proportion von_mises uniform
BENCHMARK_BINARIES = $(patsubst %,src/%,$(BENCHMARKS))
BENCHMARK_CSVS = $(patsubst %,%.csv,$(BENCHMARKS))

.PHONY : clean build

all : $(BENCHMARK_BINARIES)

clean :
	-@rm $(BENCHMARK_BINARIES)
