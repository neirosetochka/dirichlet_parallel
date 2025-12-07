#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
 
#include <cuda_runtime.h>
 
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)
 
__host__ __device__ double calcUpperBound(double x);
__host__ __device__ double calcLowerBound(double x);
__host__ __device__ double calcLeftBound(double y);
__host__ __device__ double calcRightBound(double y);
__host__ __device__ double calcIntersectionInterval(double x1, double y1, double x2, double y2, char axis);
__host__ __device__ double calcIntersectionArea(double a, double b, double y_bottom, double y_top, int steps);
__host__ __device__ double get_x_device(int i_global, double h1_val);
__host__ __device__ double get_y_device(int j_global, double h2_val);

__global__ void kernel_calc_coefficients(double* d_a, double* d_b, double* d_F,
                                         int ni, int nj, int i_start_val, int j_start_val,
                                         double h1_val, double h2_val, double eps,
                                         int M_val, int N_val, int stepsForArea);
 
__global__ void kernel_calc_D(double* d_a, double* d_b, double* d_D,
                              int ni, int nj, int i_start_val, int j_start_val,
                              double h1_val, double h2_val, int M_val, int N_val);
 
__global__ void kernel_applyA(double* d_w, double* d_result, double* d_a, double* d_b, double* d_D,
                              int ni, int nj, int i_start_val, int j_start_val,
                              double h1_val, double h2_val, int M_val, int N_val);
 
__global__ void kernel_scaled_add(double* d_A, double* d_B, double alpha, double* d_result, int ni, int nj);
 
__global__ void kernel_copy(double* d_src, double* d_dst, int ni, int nj);
 
__global__ void kernel_scalar_prod(double* d_u, double* d_v, double* d_partial, int ni, int nj, double h1_val, double h2_val);
 
__global__ void kernel_calcZ(double* d_r, double* d_D, double* d_z, int ni, int nj, double divide_eps);
 
__global__ void kernel_extract_north(double* d_u, double* d_send, int nj);
 
__global__ void kernel_extract_south(double* d_u, double* d_send, int ni, int nj);
 
__global__ void kernel_extract_west(double* d_u, double* d_send, int ni, int nj);
 
__global__ void kernel_extract_east(double* d_u, double* d_send, int ni, int nj);
 
__global__ void kernel_insert_north(double* d_u, double* d_recv, int nj);
 
__global__ void kernel_insert_south(double* d_u, double* d_recv, int ni, int nj);
 
__global__ void kernel_insert_west(double* d_u, double* d_recv, int ni, int nj);
 
__global__ void kernel_insert_east(double* d_u, double* d_recv, int ni, int nj);
 
__global__ void kernel_max_abs(double* d_r, double* d_partial, int ni, int nj);

#endif