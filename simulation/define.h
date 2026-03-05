#pragma once

#include <stdio.h>

#ifdef NOLOG
#define LOG(...)
#else
#define LOG(fmt, ...)                   \
  {                                           \
    printf("LOG: " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout);                       \
  }
#endif

#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)

#define SCALING_FACTOR_MIN 1
#define SCALING_FACTOR_MAX 1000000
#define SCALING_FACTOR_STEP 2

#define MAX_SUM_LIMIT 100
#define BENCHMARK_DIR "benchmark/"