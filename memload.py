#!/usr/bin/env python

import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cpp_source = """
std::vector<torch::Tensor> greedy_lines_allocation(torch::Tensor load_start, float decay, torch::Tensor line_requests) {
  auto nb_lines = load_start.size(1);
  auto batch_size = line_requests.size(0);
  auto nb_heads = line_requests.size(1);
  auto T = line_requests.size(2);

  auto load_start_a = load_start.accessor<float,2>();
  auto line_requests_a = line_requests.accessor<float,3>();

  auto load = torch::empty({batch_size, nb_lines, T});
  auto load_a = load.accessor<float,3>();

  auto allocation_result = torch::empty({batch_size,nb_heads,T},torch::TensorOptions().dtype(torch::kInt64));
  auto allocation_result_a = allocation_result.accessor<long,3>();

  for(int n = 0; n < batch_size; n++) {
    for(int t = 0; t < T; t++) {
      for(int l = 0; l < nb_lines; l++) {
        if(t == 0) {
          load[n][l][t] = decay * load_start_a[n][l];
        } else {
          load[n][l][t] = decay * load[n][l][t-1];
        }
      }
      for(int h = 0; h < nb_heads; h++) {
        if(line_requests_a[n][h][t] > 0) {
          int l_lowest_load;
          for(int l = 0; l < nb_lines; l++) {
            if(l == 0 || load_a[n][l][t]<load_a[n][l_lowest_load][t]) l_lowest_load=l;
          }
          if(load_a[n][l_lowest_load][t] < line_requests_a[n][h][t]) {
            allocation_result_a[n][h][t] = l_lowest_load;
            load_a[n][l_lowest_load][t] = line_requests_a[n][h][t];
          } else {
            allocation_result_a[n][h][t] = -1;
          }
        } else {
          allocation_result_a[n][h][t] = -1;
        }
      }
    }
  }

  return {allocation_result,load};
}
"""

######################################################################

allocator_module = torch.utils.cpp_extension.load_inline(
    name="allocator_module",
    cpp_sources=[cpp_source],
    functions=["greedy_lines_allocation"],
    build_directory="/tmp/",
    # verbose=True,
)

lines_allocation = allocator_module.greedy_lines_allocation

######################################################################

if __name__ == "__main__":
    N, H, L, T = 1, 1, 3, 20

    load_start = torch.rand(N, L)
    requests = (2 * torch.rand(N, H, T) - 1).clamp(min=0)

    print("load_start", load_start)

    print("requests", requests)

    alloc, load = lines_allocation(load_start, 0.99, requests)

    print("alloc", alloc)

    print("load", load)
