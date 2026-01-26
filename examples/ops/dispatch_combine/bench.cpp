#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "mori/ops/dispatch_combine/layout_transform_kernels.hpp"

#define CHECK_HIP(expr)                                                        \
  do {                                                                         \
    hipError_t _status = (expr);                                               \
    if (_status != hipSuccess) {                                               \
      std::cerr << "HIP error " << hipGetErrorString(_status)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

struct BenchConfig {
  int64_t num_tokens{128};
  int hidden_dim{7186};
  int experts{16};
  int topk{8};
  int loops{5};
};

void PrintUsage(std::string_view exe) {
  std::cout << "Usage: " << exe << " [--N tokens] [--H hidden] [--E experts] "
            << "[--K topk] [--loops repeat]" << std::endl;
}

int main(int argc, char** argv) {
  BenchConfig config;
  std::string_view prog = argc > 0 ? argv[0] : "bench";
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--N" && i + 1 < argc) {
      config.num_tokens = std::stoll(argv[++i]);
    } else if (arg == "--H" && i + 1 < argc) {
      config.hidden_dim = std::stoi(argv[++i]);
    } else if (arg == "--E" && i + 1 < argc) {
      config.experts = std::stoi(argv[++i]);
    } else if (arg == "--K" && i + 1 < argc) {
      config.topk = std::stoi(argv[++i]);
    } else if (arg == "--loops" && i + 1 < argc) {
      config.loops = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(prog);
      return 0;
    }
  }

  if (config.num_tokens <= 0 || config.hidden_dim <= 0 || config.experts <= 0 || config.topk <= 0) {
    PrintUsage(prog);
    return EXIT_FAILURE;
  }

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  const size_t max_items = static_cast<size_t>(config.num_tokens) * config.topk;
  const size_t dispatch_elements = static_cast<size_t>(config.num_tokens) * config.hidden_dim;
  const size_t packed_elements = static_cast<size_t>(config.experts) * config.num_tokens * config.hidden_dim;
  const size_t reconstruction_elements = dispatch_elements;

  std::vector<float> host_dispatch(dispatch_elements);
  std::vector<mori::moe::index_t> host_indices(max_items);

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<mori::moe::index_t> expert_dist(0, config.experts - 1);

  for (float& v : host_dispatch) {
    v = value_dist(rng);
  }
  for (mori::moe::index_t& idx : host_indices) {
    idx = expert_dist(rng);
  }

  float* d_dispatch = nullptr;
  float* d_packed = nullptr;
  float* d_recon = nullptr;
  CHECK_HIP(hipMalloc(&d_dispatch, dispatch_elements * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_packed, packed_elements * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_recon, reconstruction_elements * sizeof(float)));

  mori::moe::index_t* d_indices = nullptr;
  mori::moe::index_t* d_sorted_tokens = nullptr;
  mori::moe::index_t* d_sorted_experts = nullptr;
  mori::moe::index_t* d_slot_indices = nullptr;
  mori::moe::index_t* d_expert_counts = nullptr;
  int* d_total_valid_count = nullptr;

  CHECK_HIP(hipMalloc(&d_indices, max_items * sizeof(mori::moe::index_t)));
  CHECK_HIP(hipMalloc(&d_sorted_tokens, max_items * sizeof(mori::moe::index_t)));
  CHECK_HIP(hipMalloc(&d_sorted_experts, max_items * sizeof(mori::moe::index_t)));
  CHECK_HIP(hipMalloc(&d_slot_indices, max_items * sizeof(mori::moe::index_t)));
  CHECK_HIP(hipMalloc(&d_expert_counts, config.experts * sizeof(mori::moe::index_t)));
  CHECK_HIP(hipMalloc(&d_total_valid_count, sizeof(int)));

  CHECK_HIP(hipMemcpyAsync(d_dispatch, host_dispatch.data(), dispatch_elements * sizeof(float), hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipMemcpyAsync(d_indices, host_indices.data(), max_items * sizeof(mori::moe::index_t), hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipMemsetAsync(d_packed, 0, packed_elements * sizeof(float), stream));
  CHECK_HIP(hipMemsetAsync(d_recon, 0, reconstruction_elements * sizeof(float), stream));

  CHECK_HIP(hipStreamSynchronize(stream));

  mori::moe::LaunchPrepareTransformMetadata(
      d_indices, d_sorted_tokens, d_sorted_experts, d_slot_indices, d_expert_counts,
      d_total_valid_count, config.num_tokens, config.topk, config.experts, 0, stream);

  CHECK_HIP(hipStreamSynchronize(stream));

  int host_valid_count = 0;
  CHECK_HIP(hipMemcpy(&host_valid_count, d_total_valid_count, sizeof(int), hipMemcpyDeviceToHost));
  std::cout << "Routing settled: " << host_valid_count << " valid tokens" << std::endl;

  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));

  const int stride_src_n = config.hidden_dim;
  const int stride_src_h = 1;
  const int stride_dst_e = config.num_tokens * config.hidden_dim;
  const int stride_dst_c = config.hidden_dim;
  const int stride_dst_h = 1;
  float transform_time = 0.0f;
  for (int iter = 0; iter < config.loops; ++iter) {
    CHECK_HIP(hipMemsetAsync(d_packed, 0, packed_elements * sizeof(float), stream));
    CHECK_HIP(hipEventRecord(start, stream));
    mori::moe::LaunchTransformDispatchOutput<float>(
      d_dispatch, d_packed, nullptr, nullptr, d_sorted_tokens, d_sorted_experts, d_slot_indices,
      stride_src_n, stride_src_h, stride_dst_e, stride_dst_c, stride_dst_h,
      host_valid_count, config.hidden_dim, 0, stream, d_total_valid_count);
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float elapsed = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
    transform_time += elapsed;
  }

  float inverse_time = 0.0f;
  for (int iter = 0; iter < config.loops; ++iter) {
    CHECK_HIP(hipMemsetAsync(d_recon, 0, reconstruction_elements * sizeof(float), stream));
    CHECK_HIP(hipEventRecord(start, stream));
    const int inv_stride_src_e = config.num_tokens * config.hidden_dim;
    const int inv_stride_src_c = config.hidden_dim;
    const int inv_stride_src_h = 1;
    const int inv_stride_dst_n = config.hidden_dim;
    const int inv_stride_dst_h = 1;
    mori::moe::LaunchInverseTransformDispatchOutput<float>(
      d_packed, d_recon, d_sorted_tokens, d_sorted_experts, d_slot_indices,
      inv_stride_src_e, inv_stride_src_c, inv_stride_src_h,
      inv_stride_dst_n, inv_stride_dst_h,
      host_valid_count, config.hidden_dim, stream);
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float elapsed = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
    inverse_time += elapsed;
  }

  transform_time /= config.loops;
  inverse_time /= config.loops;
  std::cout << "Average transform kernel time: " << transform_time << " ms" << std::endl;
  std::cout << "Average inverse kernel time: " << inverse_time << " ms" << std::endl;

  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));
  CHECK_HIP(hipFree(d_dispatch));
  CHECK_HIP(hipFree(d_packed));
  CHECK_HIP(hipFree(d_recon));
  CHECK_HIP(hipFree(d_indices));
  CHECK_HIP(hipFree(d_sorted_tokens));
  CHECK_HIP(hipFree(d_sorted_experts));
  CHECK_HIP(hipFree(d_slot_indices));
  CHECK_HIP(hipFree(d_expert_counts));
  CHECK_HIP(hipFree(d_total_valid_count));
  CHECK_HIP(hipStreamDestroy(stream));

  return EXIT_SUCCESS;
}
