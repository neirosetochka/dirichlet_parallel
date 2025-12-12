#include "cuda_kernels.h"
#include <cmath>
 
__host__ __device__ double calcUpperBound(double x) { 
    return fmin(2.0 - fabs(x), 1.0); 
}
 
__host__ __device__ double calcLowerBound(double x) { 
    return fabs(x) - 2.0; 
}
 
__host__ __device__ double calcLeftBound(double y) { 
    return fabs(y) - 2.0; 
}
 
__host__ __device__ double calcRightBound(double y) { 
    return 2.0 - fabs(y); 
}
 
__host__ __device__ double calcIntersectionInterval(double x1, double y1, double x2, double y2, char axis) {
    if (axis == 'y') {
        double x = x1;
        if (x <= -2.0 || x >= 2.0) return 0.0;
        double up = calcUpperBound(x);
        double down = calcLowerBound(x);
        return fmax(fmin(y2, up) - fmax(y1, down), 0.0);
    } else if (axis == 'x') {
        double y = y1;
        if (y <= -2.0 || y >= 1.0) return 0.0;
        double left = calcLeftBound(y);
        double right = calcRightBound(y);
        return fmax(fmin(x2, right) - fmax(x1, left), 0.0);
    }
    return 0.0;
}
 
__host__ __device__ double calcIntersectionArea(double a, double b, double y_bottom, double y_top, int steps) {
    if (y_bottom >= 1.0 || a >= 2.0 || b <= -2.0 || y_top <= -2.0) {
        return 0.0;
    }
    if (a >= b || y_bottom >= y_top) {
        return 0.0;
    }
    double dx = (b - a) / steps;
    double area = 0.0;
    for (int i = 0; i < steps; ++i) {
        double x = a + (i + 0.5) * dx;
        double lower_bound = calcLowerBound(x);
        double upper_bound = calcUpperBound(x);
        double intersect_low = fmax(y_bottom, lower_bound);
        double intersect_high = fmin(y_top, upper_bound);
        if (intersect_low < intersect_high) {
            area += (intersect_high - intersect_low) * dx;
        }
    }
    return area;
}
 
__host__ __device__ double get_x_device(int i_global, double h1_val) { 
    return -2.0 + i_global * h1_val;
}
 
__host__ __device__ double get_y_device(int j_global, double h2_val) { 
    return -2.0 + j_global * h2_val; 
}
 
__global__ void kernel_calc_coefficients(double* d_a, double* d_b, double* d_F,
                                         int ni, int nj, int i_start_val, int j_start_val,
                                         double h1_val, double h2_val, double eps,
                                         int M_val, int N_val, int stepsForArea) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    int gi = i_start_val + i;
    int gj = j_start_val + j;
 
    double x = get_x_device(gi, h1_val);
    double y = get_y_device(gj, h2_val);
 
    double x_left = x - 0.5 * h1_val;
    double x_right = x + 0.5 * h1_val;
    double y_down = y - 0.5 * h2_val;
    double y_up = y + 0.5 * h2_val;
 
    double y_l = calcIntersectionInterval(x_left, y_down, x_left, y_up, 'y');
    double x_l = calcIntersectionInterval(x_left, y_down, x_right, y_down, 'x');
 
    double a_val = y_l / h2_val;
    if (a_val < 1.0) {
        a_val += (1.0 - a_val) / eps;
    }
    d_a[idx] = a_val;
 
    double b_val = x_l / h1_val;
    if (b_val < 1.0) {
        b_val += (1.0 - b_val) / eps;
    }
    d_b[idx] = b_val;
 
    if (gi == 0 || gi == M_val || gj == 0 || gj == N_val) {
        d_F[idx] = 0.0;
    } else {
        d_F[idx] = calcIntersectionArea(x_left, x_right, y_down, y_up, stepsForArea) / (h1_val * h2_val);
    }
}
 
__global__ void kernel_calc_D(double* d_a, double* d_b, double* d_D,
                              int ni, int nj, int i_start_val, int j_start_val,
                              double h1_val, double h2_val, int M_val, int N_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    int idx_ip1 = (i + 2) * (nj + 2) + (j + 1);
    int idx_jp1 = (i + 1) * (nj + 2) + (j + 2);
 
    int gi = i_start_val + i;
    int gj = j_start_val + j;
 
    if (gi == 0 || gj == 0 || gi == M_val || gj == N_val) {
        d_D[idx] = 1.0;
    } else {
        d_D[idx] = (d_a[idx_ip1] + d_a[idx]) / (h1_val * h1_val) + 
                   (d_b[idx_jp1] + d_b[idx]) / (h2_val * h2_val);
    }
}
 
__global__ void kernel_applyA(double* d_w, double* d_result, double* d_a, double* d_b, double* d_D,
                              int ni, int nj, int i_start_val, int j_start_val,
                              double h1_val, double h2_val, int M_val, int N_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    int idx_ip1 = (i + 2) * (nj + 2) + (j + 1);
    int idx_im1 = (i) * (nj + 2) + (j + 1);
    int idx_jp1 = (i + 1) * (nj + 2) + (j + 2);
    int idx_jm1 = (i + 1) * (nj + 2) + (j);
 
    int gi = i_start_val + i;
    int gj = j_start_val + j;
 
    if (gi == 0 || gj == 0 || gi == M_val || gj == N_val) {
        d_result[idx] = d_w[idx];
    } else {
        d_result[idx] = d_D[idx] * d_w[idx] -
            d_a[idx_ip1] / (h1_val * h1_val) * d_w[idx_ip1] -
            d_a[idx] / (h1_val * h1_val) * d_w[idx_im1] -
            d_b[idx_jp1] / (h2_val * h2_val) * d_w[idx_jp1] -
            d_b[idx] / (h2_val * h2_val) * d_w[idx_jm1];
    }
}
 
__global__ void kernel_scaled_add(double* d_A, double* d_B, double alpha, double* d_result, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    d_result[idx] = d_A[idx] + alpha * d_B[idx];
}
 
__global__ void kernel_copy(double* d_src, double* d_dst, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    d_dst[idx] = d_src[idx];
}
 
__global__ void kernel_scalar_prod(double* d_u, double* d_v, double* d_partial, int ni, int nj, double h1_val, double h2_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    int linear_idx = i * nj + j;
    d_partial[linear_idx] = d_u[idx] * d_v[idx] * h1_val * h2_val;
}
 
__global__ void kernel_calcZ(double* d_r, double* d_D, double* d_z, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    d_z[idx] = d_r[idx] / d_D[idx];
}
 
__global__ void kernel_extract_north(double* d_u, double* d_send, int nj) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nj) return;
    int idx = (0 + 1) * (nj + 2) + (j + 1);
    d_send[j] = d_u[idx];
}
 
__global__ void kernel_extract_south(double* d_u, double* d_send, int ni, int nj) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nj) return;
    int idx = (ni - 1 + 1) * (nj + 2) + (j + 1);
    d_send[j] = d_u[idx];
}
 
__global__ void kernel_extract_west(double* d_u, double* d_send, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;
    int idx = (i + 1) * (nj + 2) + (0 + 1);
    d_send[i] = d_u[idx];
}
 
__global__ void kernel_extract_east(double* d_u, double* d_send, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;
    int idx = (i + 1) * (nj + 2) + (nj - 1 + 1);
    d_send[i] = d_u[idx];
}
 
__global__ void kernel_insert_north(double* d_u, double* d_recv, int nj) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nj) return;
    int idx = (-1 + 1) * (nj + 2) + (j + 1);
    d_u[idx] = d_recv[j];
}
 
__global__ void kernel_insert_south(double* d_u, double* d_recv, int ni, int nj) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nj) return;
    int idx = (ni + 1) * (nj + 2) + (j + 1);
    d_u[idx] = d_recv[j];
}
 
__global__ void kernel_insert_west(double* d_u, double* d_recv, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;
    int idx = (i + 1) * (nj + 2) + (-1 + 1);
    d_u[idx] = d_recv[i];
}
 
__global__ void kernel_insert_east(double* d_u, double* d_recv, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;
    int idx = (i + 1) * (nj + 2) + (nj + 1);
    d_u[idx] = d_recv[i];
}
 
__global__ void kernel_max_abs(double* d_r, double* d_partial, int ni, int nj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (i >= ni || j >= nj) return;
 
    int idx = (i + 1) * (nj + 2) + (j + 1);
    int linear_idx = i * nj + j;
    d_partial[linear_idx] = fabs(d_r[idx]);
}

__global__ void kernel_reduce_sum(double* d_in, double* d_out, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 2;

    if (idx + 1 < (unsigned int)N) {
        d_out[tid] = d_in[idx] + d_in[idx + 1];
    } else if (idx < (unsigned int)N) {
        d_out[tid] = d_in[idx];
    }
}