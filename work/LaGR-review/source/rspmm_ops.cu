#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

const int kCoarseningFactor = 2;
const int kThreadPerBlock = 256;

template <class scalar_t>
__global__ void forward_kernel(
    const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
    const scalar_t *relation, const scalar_t *input,
    scalar_t *output, int64_t num_row, int64_t nnz, int64_t dim) {

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    int64_t *layer_ind_buf = buffer + blockDim.y * warpSize;
    col_ind_buf += threadIdx.y * warpSize;
    layer_ind_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
    scalar_t out[kCoarseningFactor];
#pragma unroll
    for (int64_t i = 0; i < 2; i++)
        out[i] = 0;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            layer_ind_buf[threadIdx.x] = layer_ind[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            int64_t layer = layer_ind_buf[offset_ptr];
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t x = relation[layer * dim + d] * input[col * dim + d];
                out[i] = out[i] + x;
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++) {
        int64_t d = d_start + i * warpSize;
        if (d >= dim)
            break;
        output[row * dim + d] = out[i];
    }
}

template <class scalar_t>
__global__ void backward_kernel(
    const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
    const scalar_t *relation, const scalar_t *input,
    const scalar_t *output, const scalar_t *output_grad,
    scalar_t *relation_grad, scalar_t *input_grad,
    int64_t num_row, int64_t nnz, int64_t dim) {

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    int64_t *layer_ind_buf = col_ind_buf + blockDim.y * warpSize;
    col_ind_buf += threadIdx.y * warpSize;
    layer_ind_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            layer_ind_buf[threadIdx.x] = layer_ind[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            int64_t layer = layer_ind_buf[offset_ptr];
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t rel = relation[layer * dim + d];
                scalar_t in = input[col * dim + d];
                scalar_t out = output[row * dim + d];
                scalar_t out_grad = output_grad[row * dim + d];
                scalar_t dx_drel = in;
                scalar_t dx_din = rel;
                atomicAdd(&relation_grad[layer * dim + d], out_grad * dx_drel);
                atomicAdd(&input_grad[col * dim + d], out_grad * dx_din);
            }
        }
        __syncwarp();
    }
}

torch::Tensor ind2ptr(const torch::Tensor &index, int size) {
    torch::Tensor num_per_index = at::zeros({size}, index.options().dtype(at::ScalarType::Int));
    num_per_index.scatter_add_(0, index, at::ones(index.sizes(), num_per_index.options()));
    num_per_index = num_per_index.toType(at::ScalarType::Long);
    torch::Tensor pointer = num_per_index.cumsum(0) - num_per_index;
    return pointer;
}

torch::Tensor forward_cuda(
    const torch::Tensor &edge_index_, const torch::Tensor &edge_type_,
    const torch::Tensor &relation_, const torch::Tensor &input_) {

    const torch::Tensor edge_index = edge_index_.contiguous();
    const torch::Tensor edge_type = edge_type_.contiguous();
    const torch::Tensor relation = relation_.contiguous();
    const torch::Tensor input = input_.contiguous();

    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);

    torch::Tensor output = at::empty({num_row, dim}, input.options());
    torch::Tensor row_ind = edge_index.select(0, 0);
    torch::Tensor row_ptr = ind2ptr(row_ind, num_row);
    torch::Tensor col_ind = edge_index.select(0, 1);
    torch::Tensor layer_ind = edge_type;

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32;
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_cuda", [&] {
        const int memory_size = kThreadPerBlock * (sizeof(int64_t) * 2 + sizeof(scalar_t));
        forward_kernel<scalar_t>
            <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            layer_ind.data_ptr<int64_t>(),
            relation.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_row, nnz, dim
        );
    });

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> backward_cuda(
        const torch::Tensor &edge_index_, const torch::Tensor &edge_type_,
        const torch::Tensor &relation_, const torch::Tensor &input_,
        const torch::Tensor &output_, const torch::Tensor &output_grad_) {

    const torch::Tensor edge_index = edge_index_.contiguous();
    const torch::Tensor edge_type = edge_type_.contiguous();
    const torch::Tensor relation = relation_.contiguous();
    const torch::Tensor input = input_.contiguous();
    const torch::Tensor output = output_.contiguous();
    const torch::Tensor output_grad = output_grad_.contiguous();

    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);

    torch::Tensor relation_grad = at::zeros_like(relation);
    torch::Tensor input_grad = at::zeros_like(input);
    torch::Tensor row_ind = edge_index.select(0, 0);
    torch::Tensor row_ptr = ind2ptr(row_ind, num_row);
    torch::Tensor col_ind = edge_index.select(0, 1);
    torch::Tensor layer_ind = edge_type;

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32;
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "backward_cuda", [&] {
        const int memory_size = kThreadPerBlock * (sizeof(int64_t) * 2 + sizeof(scalar_t));
        backward_kernel<scalar_t>
            <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            layer_ind.data_ptr<int64_t>(),
            relation.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            output_grad.data_ptr<scalar_t>(),
            relation_grad.data_ptr<scalar_t>(),
            input_grad.data_ptr<scalar_t>(),
            num_row, nnz, dim
        );
    });

    return std::make_tuple(relation_grad, input_grad);
}