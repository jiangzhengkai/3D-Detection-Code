#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

#define CUDA_1D_KERNEL_LOOP(i, n)           \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;    \
    i += blockDim.x * gridDim.x)


template <typename scalar_t>
__global__ void align_feature_forward_kernel(const int nthreads,
                                             const scalar_t *data,
                                             const scalar_t *weight,
                                             const int weight_height,
                                             const int weight_width,
                                             const int N,
                                             const int C,
                                             const int Size_Weight,
                                             const int H,
                                             const int W,
                                             scalar_t *output) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int w = i % W;
    const int h = (i / W) % H;
    const int c = (i / (H * W)) % C;
    const int n = (i / (C * H * W));

    int p_h = 0,p_w = 0, p_weight = 0;
    for (int i =0; i < weight_height; i++) {
      for (int j =0; j < weight_width; j++) {
        p_h = (i - weight_height/2) + h;
        p_w = (j - weight_width/2) + w;
        if (p_h >=0 && p_w >=0 && p_h < H && p_w < W) {
          p_weight = i * weight_width + j;
          int data_index = n*C*H*W + c*H*W + h*W + w;
          int weight_index = n*Size_Weight*H*W + p_weight*H*W + h*W +w;
          output[i] += data[data_index] * weight[weight_index];
          }
        }
      }
  }
}


template <typename scalar_t>
__global__ void align_feature_backward_kernel(const int nthreads,
                                              const scalar_t *grad_top,
                                              const scalar_t *data,
                                              const scalar_t *weight,
                                              const int weight_height,
                                              const int weight_width,
                                              const int N,
                                              const int C,
                                              const int Size_Weight,
                                              const int H,
                                              const int W,
                                              scalar_t *grad_data,
                                              scalar_t *grad_weight) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int w = i % W;
    const int h = (i / W) % H;
    const int c = (i / (H * W)) % C;
    const int n = (i / (C * H * W));

    int p_h = 0,p_w = 0, p_weight = 0;
    for (int i =0; i < weight_height; i++) {
      for (int j =0; j < weight_width; j++) {
        p_h = (i - weight_height/2) + h;
        p_w = (j - weight_width/2) + w;
        if (p_h >=0 && p_w >=0 && p_h < H && p_w < W) {
          p_weight = i * weight_width + j;
          int data_index = n*C*H*W + c*H*W + p_h*W + p_w;
          int weight_index = n*Size_Weight*H*w + p_weight*H*W + h*W + w;
          atomicAdd(grad_weight+weight_index, grad_top[i]*data[data_index]);
          atomicAdd(grad_data+data_index, grad_top[i]*weight[weight_index]);
          }
        }
      }
  }
}



int align_feature_cuda_forward_launcher(const at::Tensor data,
                                        const at::Tensor weight,
                                        const int weight_height,
                                        const int weight_width,
                                        const int N,
                                        const int C,
                                        const int Size_Weight,
                                        const int H,
                                        const int W,
                                        at::Tensor output) {
    AT_ASSERTM(data.dim() == 4, "data should be 4 dimensions");
    AT_ASSERTM(weight.dim() == 4, "weight should be 4 dimensions");
    AT_ASSERTM(output.dim() == 4, "output should be 4 dimensions");

    long size = N * C * H * W;
    dim3 grid(std::min(THCCeilDiv(size, 512L), 4096L));
    dim3 block(512);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data.type(), "align feature forward", ([&] {
           align_feature_forward_kernel<scalar_t><<<grid, block>>>(
               size, data.data<scalar_t>(), weight.data<scalar_t>(), weight_height, weight_width,
               N, C, Size_Weight, H, W, output.data<scalar_t>());
        }));
    THCudaCheck(cudaGetLastError());
    return 1;
}


int align_feature_cuda_backward_launcher(at::Tensor top_grad,
                                         at::Tensor data,
                                         at::Tensor weight,
                                         const int weight_height,
                                         const int weight_width,
                                         const int N,
                                         const int C,
                                         const int Size_Weight,
                                         const int H,
                                         const int W,
                                         at::Tensor grad_data,
                                         at::Tensor grad_weight) {
                                         
    AT_ASSERTM(data.dim() == 4, "data should be 4 dimensions");
    AT_ASSERTM(weight.dim() == 4, "weight should be 4 dimensions");
    AT_ASSERTM(top_grad.dim() == 4, "output should be 4 dimensions");
    AT_ASSERTM(grad_data.dim() == 4, "data shoud be 4 dimensions");
    AT_ASSERTM(grad_weight.dim() == 4, "weight should be 4 dimensions");

    long size = N * C * H * W;
    dim3 grid(std::min(THCCeilDiv(size, 512L), 4096L));
    dim3 block(512);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data.type(), "align feature backward", ([&] {
           align_feature_backward_kernel<scalar_t><<<grid, block>>>(
               size, top_grad.data<scalar_t>(), data.data<scalar_t>(), weight.data<scalar_t>(), weight_height, weight_width,
               N, C, Size_Weight, H, W, grad_data.data<scalar_t>(), grad_weight.data<scalar_t>());
        }));
    return 1;
}
