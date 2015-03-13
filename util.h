#pragma once

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>

// Container hack
#define RANGE(x) ((x).begin()), ((x).end())
#define SUM(x)   (std::accumulate(RANGE(x), .0))

// Random hack
#include <chrono>
#include <random>
#define CLOCK (std::chrono::system_clock::now().time_since_epoch().count())
static std::mt19937 _rng(CLOCK);
static std::uniform_real_distribution<double> _unif01;
static std::normal_distribution<double> _stdnormal;
//#define UNIF01  (_unif01(_rng))
static int _jxr = CLOCK;
static int xorshift_rand() {
  _jxr ^= (_jxr << 13); _jxr ^= (_jxr >> 17); _jxr ^= (_jxr << 5);
  return _jxr & 0x7fffffff;
}
#define UNIF01  (xorshift_rand() * 4.6566125e-10)
#define DICE(x) ((int)(UNIF01 * (x)))

// Google Log hack
#define LI LOG(INFO)
#define LW LOG(WARNING)
#define LD DLOG(INFO)

// Eigen
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#define EIGEN_DEFAULT_IO_FORMAT \
        Eigen::IOFormat(FullPrecision,1," "," ","","","[","]")
#include <Eigen/Dense>
using real = double;
using Eigen::Dynamic; using Eigen::ColMajor;
using EMatrix = Eigen::Matrix<real, Dynamic, Dynamic, ColMajor>;
using EVector = Eigen::Matrix<real, Dynamic, 1,       ColMajor>;
using EMAtrix = Eigen::Array <real, Dynamic, Dynamic, ColMajor>;
using EArray  = Eigen::Array <real, Dynamic, 1,       ColMajor>;
using IMatrix = Eigen::Matrix<int,  Dynamic, Dynamic, ColMajor>;
using IVector = Eigen::Matrix<int,  Dynamic, 1,       ColMajor>;
using IMAtrix = Eigen::Array <int,  Dynamic, Dynamic, ColMajor>;
using IArray  = Eigen::Array <int,  Dynamic, 1,       ColMajor>;

// Reference: Wikipedia Inverse Gaussian entry
static real draw_invgaussian(real mean, real shape) {
  real z = _stdnormal(_rng);
  real y = z * z;
  real meany = mean * y;
  real shape2 = 2.0 * shape;
  real x = mean
           + (mean * meany) / shape2
           - (mean / shape2) * sqrt(meany * (4.0 * shape + meany));
  real test = _unif01(_rng);
  return (test <= mean / (mean + x)) ? x : (mean * mean) / x;
}

// Multivariate Normal with mean = covariance * v; covariance = inv(precision)
static EVector draw_mvgaussian(const EMatrix& precision, const EVector& v) {
  Eigen::LLT<EMatrix> chol;
  chol.compute(precision);
  CHECK_EQ(chol.info(), Eigen::ComputationInfo::Success);
  EVector alpha = chol.matrixL().solve(v); // alpha = L' * mu
  EVector mu = chol.matrixU().solve(alpha);
  EVector z(v.size());
  for (int i = 0; i < v.size(); ++i)
    z(i) = _stdnormal(_rng);
  EVector x = chol.matrixU().solve(z);
  return mu + x;
}

// Draw from discrete distribution. (aka multinomial)
static int draw_discrete(const EArray& prob) {
  int sample = prob.size() - 1;
  double dart = _unif01(_rng) * prob.sum();
  for (int k = 0; k < prob.size() - 1; ++k) {
    dart -= prob(k);
    if (dart <= .0) { sample = k; break; }
  }
  CHECK_GE(sample, 0);
  CHECK_LT(sample, prob.size());
  return sample;
}

// Get monotonic time in seconds from a starting point
static double get_time() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (start.tv_sec + start.tv_nsec/1000000000.0);
}

class Timer {
public:
  void   tic() { start_ = get_time(); }
  double toc() { double ret = get_time() - start_; time_ += ret; return ret; }
  double get() { return time_; }
private:
  double time_  = .0;
  double start_ = get_time();
};

// Google flags hack
static void print_help() {
  fprintf(stderr, "Program Flags:\n");
  std::vector<google::CommandLineFlagInfo> all_flags;
  google::GetAllFlags(&all_flags);
  for (const auto& flag : all_flags) {
    if (flag.filename.find("src/") != 0) // HACK: filter out built-in flags
      fprintf(stderr,
              "-%s: %s (%s, default:%s)\n",
              flag.name.c_str(),
              flag.description.c_str(),
              flag.type.c_str(),
              flag.default_value.c_str());
  }
  exit(1);
}

// Google flags hack
static void print_flags() {
  LI << "---------------------------------------------------------------------";
  std::vector<google::CommandLineFlagInfo> all_flags;
  google::GetAllFlags(&all_flags);
  for (const auto& flag : all_flags) {
    if (flag.filename.find("src/") != 0) // HACK: filter out built-in flags
      LI << flag.name << ": " << flag.current_value;
  }
  LI << "---------------------------------------------------------------------";
}

// Faster base 10 strtol without error checking
static long int strtol(const char *nptr, char **endptr) {
  while (isspace(*nptr)) ++nptr; // skip spaces
  bool is_negative = false;
  if (*nptr == '-') { is_negative = true; ++nptr; }
  else if (*nptr == '+') { ++nptr; } // end of sign
  long int res = 0;
  while (isdigit(*nptr)) { res = (res * 10) + (*nptr - '0'); ++nptr; }
  if (endptr != NULL) *endptr = (char *)nptr;
  if (is_negative) return -res;
  return res;
}
