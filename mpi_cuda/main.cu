#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mpi.h>
#include <cuda_runtime.h>
#include "cuda_kernels.h"
 
using namespace std;
 
int M = 40, N = 40;
int mpi_rank, mpi_size;
int px, py, coords[2];
int i_start, i_end, j_start, j_end;
int local_M, local_N;
int north, south, east, west;
MPI_Comm cart_comm;
double h1, h2;
 
class TimeLogger {
private:
    std::unordered_map<std::string, double> start_times;
    std::unordered_map<std::string, double> accumulated_times;

    double now() {
        return MPI_Wtime();
    }

public:
    void start(const std::string& label) {
        start_times[label] = now();
    }

    void stop(const std::string& label) {
        double t_end = now();
        if (start_times.find(label) != start_times.end()) {
            accumulated_times[label] += t_end - start_times[label];
        }
    }
 
    void report(int mpi_rank, int mpi_size, MPI_Comm comm) const {
        if (mpi_rank == 0) {
            cout << "\n====== TIME REPORT ======\n";
        }
        for (const auto& kv: accumulated_times) {
            const string& label = kv.first;
            double local_time = kv.second;
            double sum_time = 0.0;
            MPI_Reduce(&local_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
 
            if (mpi_rank == 0) {
                double avg_time = sum_time / mpi_size;
                cout << " - " << label << ": " << avg_time << "s\n";
            }
        }
        if (mpi_rank == 0) {
            cout << "=========================\n\n";
        }
    }
};
TimeLogger timer;
 
class LocalGridFunctionGPU {
public:
    double* d_data;
    int ni, nj;
    int total_size;
 
    LocalGridFunctionGPU(int m, int n) : ni(m), nj(n) {
        timer.start("GPU Memory Management");
        total_size = (m + 2) * (n + 2);
        CUDA_CHECK(cudaMalloc(&d_data, total_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_data, 0, total_size * sizeof(double)));
        timer.stop("GPU Memory Management");
    }
 
    ~LocalGridFunctionGPU() {
        if (d_data) {
            timer.start("GPU Memory Management");
            cudaFree(d_data);
            timer.stop("GPU Memory Management");
            d_data = NULL;
        }
    }
 
    void copyToHost(vector<double>& host_data) const {
        host_data.resize(total_size);
        timer.start("GPU to Host Copy");
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_data, total_size * sizeof(double), cudaMemcpyDeviceToHost));
        timer.stop("GPU to Host Copy");
    }
 
    void copyFromHost(const vector<double>& host_data) {
        timer.start("Host to GPU Copy");
        CUDA_CHECK(cudaMemcpy(d_data, host_data.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
        timer.stop("Host to GPU Copy");
    }
};
 
void compute_2d_decomposition() {
    double min_dist = 1e10;
    px = 1;
    py = mpi_size;
 
    double p_ideal = sqrt(mpi_size * (M + 1.0) / (N + 1.0));
    int p_min = ceil(sqrt(mpi_size * (M + 1.0) / (N + 1.0) / 2.0));
    int p_max = floor(sqrt(2.0 * mpi_size * (M + 1.0) / (N + 1.0)));
    for (int p = p_min; p <= min(p_max, mpi_size); ++p) {
        if (mpi_size % p == 0) {
            double dist = fabs(p - p_ideal);
            if (dist < min_dist) {
                px = p;
                py = mpi_size / p;
                min_dist = dist;
            }
        }
    }
}
 
void setup_2d_domain() {
    compute_2d_decomposition();
    int dims[2] = {px, py};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, mpi_rank, 2, coords);
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
 
    int base_i = (M + 1) / px;
    int rem_i = (M + 1) % px;
    if (coords[0] < rem_i) {
        local_M = base_i + 1;
        i_start = coords[0] * (base_i + 1);
    } else {
        local_M = base_i;
        i_start = rem_i * (base_i + 1) + (coords[0] - rem_i) * base_i;
    }
    i_end = i_start + local_M - 1;
 
    int base_j = (N + 1) / py;
    int rem_j = (N + 1) % py;
    if (coords[1] < rem_j) {
        local_N = base_j + 1;
        j_start = coords[1] * (base_j + 1);
    } else {
        local_N = base_j;
        j_start = rem_j * (base_j + 1) + (coords[1] - rem_j) * base_j;
    }
    j_end = j_start + local_N - 1;
 
    h1 = 4.0 / M;
    h2 = 3.0 / N;
}
 
void exchange_ghosts_gpu(LocalGridFunctionGPU& u) {
    int blockSize = 256;
 
    double *d_send_north, *d_send_south, *d_send_west, *d_send_east;
    double *d_recv_north, *d_recv_south, *d_recv_west, *d_recv_east;
 
    timer.start("GPU Memory Management");
    CUDA_CHECK(cudaMalloc(&d_send_north, local_N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_send_south, local_N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_send_west, local_M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_send_east, local_M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv_north, local_N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv_south, local_N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv_west, local_M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv_east, local_M * sizeof(double)));
    timer.stop("GPU Memory Management");
 
    int gridN = (local_N + blockSize - 1) / blockSize;
    int gridM = (local_M + blockSize - 1) / blockSize;
 
    timer.start("GPU Kernels");
    kernel_extract_north<<<gridN, blockSize>>>(u.d_data, d_send_north, local_N);
    kernel_extract_south<<<gridN, blockSize>>>(u.d_data, d_send_south, local_M, local_N);
    kernel_extract_west<<<gridM, blockSize>>>(u.d_data, d_send_west, local_M, local_N);
    kernel_extract_east<<<gridM, blockSize>>>(u.d_data, d_send_east, local_M, local_N);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop("GPU Kernels");
 
    vector<double> h_send_north(local_N), h_send_south(local_N);
    vector<double> h_send_west(local_M), h_send_east(local_M);
    vector<double> h_recv_north(local_N), h_recv_south(local_N);
    vector<double> h_recv_west(local_M), h_recv_east(local_M);
 
    timer.start("GPU to Host Copy");
    CUDA_CHECK(cudaMemcpy(h_send_north.data(), d_send_north, local_N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_south.data(), d_send_south, local_N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_west.data(), d_send_west, local_M * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_east.data(), d_send_east, local_M * sizeof(double), cudaMemcpyDeviceToHost));
    timer.stop("GPU to Host Copy");
 
    MPI_Request reqs[8];
    int nreq = 0;
 
    timer.start("MPI Communication");
    if (north != MPI_PROC_NULL) {
        MPI_Isend(h_send_north.data(), local_N, MPI_DOUBLE, north, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(h_recv_north.data(), local_N, MPI_DOUBLE, north, 1, cart_comm, &reqs[nreq++]);
    }
    if (south != MPI_PROC_NULL) {
        MPI_Isend(h_send_south.data(), local_N, MPI_DOUBLE, south, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(h_recv_south.data(), local_N, MPI_DOUBLE, south, 0, cart_comm, &reqs[nreq++]);
    }
    if (west != MPI_PROC_NULL) {
        MPI_Isend(h_send_west.data(), local_M, MPI_DOUBLE, west, 2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(h_recv_west.data(), local_M, MPI_DOUBLE, west, 3, cart_comm, &reqs[nreq++]);
    }
    if (east != MPI_PROC_NULL) {
        MPI_Isend(h_send_east.data(), local_M, MPI_DOUBLE, east, 3, cart_comm, &reqs[nreq++]);
        MPI_Irecv(h_recv_east.data(), local_M, MPI_DOUBLE, east, 2, cart_comm, &reqs[nreq++]);
    }
 
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
    timer.stop("MPI Communication");
 
    timer.start("Host to GPU Copy");
    if (north != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_recv_north, h_recv_north.data(), local_N * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (south != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_recv_south, h_recv_south.data(), local_N * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (west != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_recv_west, h_recv_west.data(), local_M * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (east != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_recv_east, h_recv_east.data(), local_M * sizeof(double), cudaMemcpyHostToDevice));
    }
    timer.stop("Host to GPU Copy");
 
    timer.start("GPU Kernels");
    if (north != MPI_PROC_NULL) {
        kernel_insert_north<<<gridN, blockSize>>>(u.d_data, d_recv_north, local_N);
    }
    if (south != MPI_PROC_NULL) {
        kernel_insert_south<<<gridN, blockSize>>>(u.d_data, d_recv_south, local_M, local_N);
    }
    if (west != MPI_PROC_NULL) {
        kernel_insert_west<<<gridM, blockSize>>>(u.d_data, d_recv_west, local_M, local_N);
    }
    if (east != MPI_PROC_NULL) {
        kernel_insert_east<<<gridM, blockSize>>>(u.d_data, d_recv_east, local_M, local_N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop("GPU Kernels");
 
    timer.start("GPU Memory Management");
    cudaFree(d_send_north);
    cudaFree(d_send_south);
    cudaFree(d_send_west);
    cudaFree(d_send_east);
    cudaFree(d_recv_north);
    cudaFree(d_recv_south);
    cudaFree(d_recv_west);
    cudaFree(d_recv_east);
    timer.stop("GPU Memory Management");
}
 
double calcScalarProdGPU(LocalGridFunctionGPU& u, LocalGridFunctionGPU& v, double* d_partial, double* h_partial) {
    dim3 blockDim(16, 16);
    dim3 gridDim((local_M + blockDim.x - 1) / blockDim.x, (local_N + blockDim.y - 1) / blockDim.y);
 
    timer.start("calcScalarProd cycle");
    timer.start("GPU Kernels");
    kernel_scalar_prod<<<gridDim, blockDim>>>(u.d_data, v.d_data, d_partial, local_M, local_N, h1, h2);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop("GPU Kernels");
    timer.stop("calcScalarProd cycle");
 
    int n_elements = local_M * local_N;
    
    timer.start("GPU to Host Copy");
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, n_elements * sizeof(double), cudaMemcpyDeviceToHost));
    timer.stop("GPU to Host Copy");
 
    double local_sum = 0.0;
    for (int i = 0; i < n_elements; ++i) {
        local_sum += h_partial[i];
    }
 
    timer.start("MPI Communication");
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    timer.stop("MPI Communication");
    
    return global_sum;
}
 
void calcScaledAddGPU(LocalGridFunctionGPU& A, LocalGridFunctionGPU& B, double alpha, LocalGridFunctionGPU& result) {
    dim3 blockDim(16, 16);
    dim3 gridDim((local_M + blockDim.x - 1) / blockDim.x, (local_N + blockDim.y - 1) / blockDim.y);
 
    timer.start("calcScaledAdd cycle");
    timer.start("GPU Kernels");
    if (alpha == 0.0) {
        if (A.d_data != result.d_data) {
            kernel_copy<<<gridDim, blockDim>>>(A.d_data, result.d_data, local_M, local_N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } else {
        kernel_scaled_add<<<gridDim, blockDim>>>(A.d_data, B.d_data, alpha, result.d_data, local_M, local_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop("GPU Kernels");
    timer.stop("calcScaledAdd cycle");
}
 
struct CoefficientsGPU {
    LocalGridFunctionGPU a;
    LocalGridFunctionGPU b;
    LocalGridFunctionGPU F;
    LocalGridFunctionGPU D;
    int stepsForArea;
 
    CoefficientsGPU(int steps) : stepsForArea(steps), a(local_M, local_N), b(local_M, local_N),
        F(local_M, local_N), D(local_M, local_N){}
 
    void calcCoefficients() {
        double eps = max(h1, h2) * max(h1, h2);
 
        dim3 blockDim(16, 16);
        dim3 gridDim((local_M + blockDim.x - 1) / blockDim.x, (local_N + blockDim.y - 1) / blockDim.y);
 
        timer.start("a, b, F calculation cycle");
        timer.start("GPU Kernels");
        kernel_calc_coefficients<<<gridDim, blockDim>>>(a.d_data, b.d_data, F.d_data,
                                                        local_M, local_N, i_start, j_start,
                                                        h1, h2, eps, M, N, stepsForArea);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop("GPU Kernels");
        timer.stop("a, b, F calculation cycle");
 
        exchange_ghosts_gpu(a);
        exchange_ghosts_gpu(b);
 
        timer.start("D calculation cycle");
        timer.start("GPU Kernels");
        kernel_calc_D<<<gridDim, blockDim>>>(a.d_data, b.d_data, D.d_data,
                                             local_M, local_N, i_start, j_start,
                                             h1, h2, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop("GPU Kernels");
        timer.stop("D calculation cycle");
    }
 
    void applyA(LocalGridFunctionGPU& w, LocalGridFunctionGPU& result) {
        
        exchange_ghosts_gpu(w);
 
        dim3 blockDim(16, 16);
        dim3 gridDim((local_M + blockDim.x - 1) / blockDim.x, (local_N + blockDim.y - 1) / blockDim.y);
 
        timer.start("applyA cycle");
        timer.start("GPU Kernels");
        kernel_applyA<<<gridDim, blockDim>>>(w.d_data, result.d_data, a.d_data, b.d_data, D.d_data,
                                             local_M, local_N, i_start, j_start, h1, h2, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop("GPU Kernels");
        timer.stop("applyA cycle");
    }
};
 
struct SolverGPU {
    LocalGridFunctionGPU w;
    LocalGridFunctionGPU p;
    LocalGridFunctionGPU r;
    LocalGridFunctionGPU z;
    int max_steps;
    int steps;
    double delta;
    double divide_eps;
 
    double* d_partial;
    double* h_partial;
 
    SolverGPU(int max_steps_val, double delta_val, double divide_eps_val = 1e-10) : 
        max_steps(max_steps_val), delta(delta_val), divide_eps(divide_eps_val),
        w(local_M, local_N), p(local_M, local_N), r(local_M, local_N), z(local_M, local_N) {
        timer.start("GPU Memory Management");
        int n_elements = local_M * local_N;
        CUDA_CHECK(cudaMalloc(&d_partial, n_elements * sizeof(double)));
        timer.stop("GPU Memory Management");
        h_partial = new double[n_elements];
    }
 
    ~SolverGPU() {
        timer.start("GPU Memory Management");
        cudaFree(d_partial);
        timer.stop("GPU Memory Management");
        delete[] h_partial;
    }
 
    void calcZ(const CoefficientsGPU& coef) {
        dim3 blockDim(16, 16);
        dim3 gridDim((local_M + blockDim.x - 1) / blockDim.x, (local_N + blockDim.y - 1) / blockDim.y);
 
        timer.start("calcZ cycle");
        timer.start("GPU Kernels");
        kernel_calcZ<<<gridDim, blockDim>>>(r.d_data, coef.D.d_data, z.d_data, local_M, local_N, divide_eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop("GPU Kernels");
        timer.stop("calcZ cycle");
    }
 
    void solve(CoefficientsGPU& coef) {
        LocalGridFunctionGPU Aw(local_M, local_N);
        LocalGridFunctionGPU& Ap = Aw;
 
        coef.applyA(w, Aw);
        calcScaledAddGPU(coef.F, Aw, -1.0, r);
        calcZ(coef);
        calcScaledAddGPU(z, p, 0.0, p);
        coef.applyA(p, Ap);
 
        double zr_scalar_prod = calcScalarProdGPU(z, r, d_partial, h_partial);
        double zr_scalar_prod_prev = zr_scalar_prod;
        double alpha = zr_scalar_prod / (calcScalarProdGPU(Ap, p, d_partial, h_partial) + divide_eps);
 
        int k;
        for (k = 1; k < max_steps; k++) {
            calcScaledAddGPU(w, p, alpha, w);
            calcScaledAddGPU(r, Ap, -alpha, r);
            double norm = fabs(alpha) * sqrt(calcScalarProdGPU(p, p, d_partial, h_partial));
            if (norm < delta) {
                steps = k;
                break;
            }
            calcZ(coef);
            zr_scalar_prod = calcScalarProdGPU(z, r, d_partial, h_partial);
            double beta = zr_scalar_prod / (zr_scalar_prod_prev + divide_eps);
            zr_scalar_prod_prev = zr_scalar_prod;
            calcScaledAddGPU(z, p, beta, p);
            coef.applyA(p, Ap);
            alpha = zr_scalar_prod / (calcScalarProdGPU(Ap, p, d_partial, h_partial) + divide_eps);
            if (k == max_steps - 1) {
                steps = max_steps;
            }
        }
    }
 
    void cout_discrepancy() {
        dim3 blockDim(16, 16);
        dim3 gridDim((local_M + blockDim.x - 1) / blockDim.x, (local_N + blockDim.y - 1) / blockDim.y);
 
        kernel_max_abs<<<gridDim, blockDim>>>(r.d_data, d_partial, local_M, local_N);
        CUDA_CHECK(cudaDeviceSynchronize());
 
        int n_elements = local_M * local_N;
        
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, n_elements * sizeof(double), cudaMemcpyDeviceToHost));
 
        double local_max_disc = 0.0;
        for (int i = 0; i < n_elements; ++i) {
            local_max_disc = max(local_max_disc, h_partial[i]);
        }
 
        double global_max_disc;
        MPI_Reduce(&local_max_disc, &global_max_disc, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
 
        if (mpi_rank == 0) {
            cout << "Solution discrepancy: " << global_max_disc << endl;
        }
    }
};
 
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    timer.start("Total time");
    timer.start("Initialization");
    
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(mpi_rank % num_devices));
    }
 
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }
    if (mpi_rank == 0) {
        printf("\n\nGrid size: M=%d, N=%d\n", M, N);
    }

    setup_2d_domain();
    timer.stop("Initialization");
 
    CoefficientsGPU coef(10);
    coef.calcCoefficients();
 
    SolverGPU solver((M - 1) * (N - 1), 1e-20, 1e-40);
    solver.solve(coef);

    timer.stop("Total time");
    
    if (mpi_rank == 0) {
        cout << "Iteration number: " << solver.steps << endl;
        cout << "MPI processes: " << mpi_size << endl;
    }
    
    solver.cout_discrepancy();
    timer.report(mpi_rank, mpi_size, MPI_COMM_WORLD);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}