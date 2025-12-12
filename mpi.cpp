#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mpi.h>

using namespace std;
 
// (M + 1) и (N + 1) узлов по оси x и y соответственно
// x - индекс i, y - индекс j
int M = 40, N = 40;
 
int mpi_rank, mpi_size; // номер процесса и количество процессов
int px, py, coords[2]; // размер сетки (в доменах) и координаты процесса в ней
int i_start, i_end, j_start, j_end; // индексы домена (без ghost)
int local_M, local_N; // количество узлов в домене (без ghost)
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

double calcUpperBound(double x) { return fmin(2 - fabs(x), 1); }
double calcLowerBound(double x) { return fabs(x) - 2; }
double calcLeftBound(double y) { return fabs(y) - 2; }
double calcRightBound(double y) { return 2 - fabs(y); }
 
double get_x(int i_global) { 
    return -2.0 + i_global * h1;
}
 
double get_y(int j_global) { 
    return -2.0 + j_global * h2; 
}

double calcIntersectionInterval(double x1, double y1, double x2, double y2, char axis) {
    if (axis == 'y') {
        double x = x1;
        if (x <= -2 || x >= 2) return 0.0;
        double up = calcUpperBound(x), down = calcLowerBound(x);
        return fmax(fmin(y2, up) - fmax(y1, down), 0.0);
    } else if (axis == 'x') {
        double y = y1;
        if (y <= -2 || y >= 1) return 0.0;
        double left = calcLeftBound(y), right = calcRightBound(y);
        return fmax(fmin(x2, right) - fmax(x1, left), 0.0);
    }
    return 0.0;
}
 
double calcIntersectionArea(double a, double b, double y_bottom, double y_top, int steps) {
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

class LocalGridFunction {
public:
    double* data;
    int ni, nj;
 
    LocalGridFunction(int m, int n) : ni(m), nj(n) {
        // +1 для ghost: реальный индекс сдвигается на +1
        // ghosts: i = -1 и j = -1
        data = new double[(m+2)*(n+2)];
        for(int i=0; i<=(m+2)*(n+2)-1;i++) data[i]=0.0;
    }
 
    double& operator()(int i, int j) {
        // индексация с учетом ghosts
        return data[(i + 1) * (nj + 2) + (j + 1)];
    }
 
    const double& operator()(int i, int j) const {
        return data[(i + 1) * (nj + 2) + (j + 1)];
    }
};
 
void compute_2d_decomposition() {
    // разбиение по алгоритму из отчета
    double min_dist = 1e10;
    px = 1;
    py = mpi_size;

    double p_ideal = sqrt(mpi_size * (M + 1) / (N + 1));
    int p_min = ceil(sqrt(mpi_size * (M + 1) / (N + 1) / 2));
    int p_max = floor(sqrt(2 * mpi_size * (M + 1) / (N + 1)));
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
    // сначала распределяем остаток
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
 
void exchange_ghosts(LocalGridFunction& u) {
    MPI_Request reqs[8];
    int nreq = 0;
 
    vector<double> send_west(local_M), send_east(local_M);
    vector<double> send_south(local_N), send_north(local_N);
    vector<double> recv_west(local_M), recv_east(local_M);
    vector<double> recv_south(local_N), recv_north(local_N);
 
    for (int j = 0; j < local_N; ++j) {
        send_north[j] = u(0, j);
        send_south[j] = u(local_M - 1, j);
    }
    for (int i = 0; i < local_M; ++i) {
        send_west[i] = u(i, 0);
        send_east[i] = u(i, local_N - 1);
    }
 
    timer.start("MPI Communication");
    if (north != MPI_PROC_NULL) {
        MPI_Isend(send_north.data(), local_N, MPI_DOUBLE, north, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_north.data(), local_N, MPI_DOUBLE, north, 1, cart_comm, &reqs[nreq++]);
    }
    if (south != MPI_PROC_NULL) {
        MPI_Isend(send_south.data(), local_N, MPI_DOUBLE, south, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_south.data(), local_N, MPI_DOUBLE, south, 0, cart_comm, &reqs[nreq++]);
    }
    if (west != MPI_PROC_NULL) {
        MPI_Isend(send_west.data(), local_M, MPI_DOUBLE, west, 2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_west.data(), local_M, MPI_DOUBLE, west, 3, cart_comm, &reqs[nreq++]);
    }
    if (east != MPI_PROC_NULL) {
        MPI_Isend(send_east.data(), local_M, MPI_DOUBLE, east, 3, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_east.data(), local_M, MPI_DOUBLE, east, 2, cart_comm, &reqs[nreq++]);
    }
 
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
    timer.stop("MPI Communication");
 
    if (north != MPI_PROC_NULL) {
        for (int j = 0; j < local_N; ++j)
            u(-1, j) = recv_north[j];
    }
    if (south != MPI_PROC_NULL) {
        for (int j = 0; j < local_N; ++j)
            u(local_M, j) = recv_south[j];
    }
    if (west != MPI_PROC_NULL) {
        for (int i = 0; i < local_M; ++i)
            u(i, -1) = recv_west[i];
    }
    if (east != MPI_PROC_NULL) {
        for (int i = 0; i < local_M; ++i)
            u(i, local_N) = recv_east[i];
    }
}
 
double calcScalarProd(const LocalGridFunction& u, const LocalGridFunction& v) {
    timer.start("calcScalarProd cycle");
    double local_sum = 0.0;
    for (int i = 0; i < local_M; ++i) {
        for (int j = 0; j < local_N; ++j) {
            local_sum += u(i, j) * v(i, j) * h1 * h2;
        }
    }
    double global_sum;
    timer.stop("calcScalarProd cycle");

    timer.start("MPI Communication");
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    timer.stop("MPI Communication");
    return global_sum;
}
 
void calcScaledAdd(const LocalGridFunction& A, const LocalGridFunction& B, double alpha, LocalGridFunction& result) {
    timer.start("calcScaledAdd cycle");
    if (alpha == 0.0) {
        const bool resultIsA = (&A == &result);
        if (!resultIsA) {
            for (int i = 0; i < local_M; ++i) {
                for (int j = 0; j < local_N; ++j) {
                    result(i, j) = A(i, j);
                }
            }
        }
    } else {
        for (int i = 0; i < local_M; ++i) {
            for (int j = 0; j < local_N; ++j) {
                result(i, j) = A(i, j) + alpha * B(i, j);
            }
        }
    }
    timer.stop("calcScaledAdd cycle");
}
 
struct Coefficients {
    LocalGridFunction a;
    LocalGridFunction b;
    LocalGridFunction F;
    LocalGridFunction D;
    int stepsForArea;
 
    Coefficients(int steps) : stepsForArea(steps), a(local_M, local_N), b(local_M, local_N),
        F(local_M, local_N), D(local_M, local_N){}
 
    void calcCoefficients() {
        double eps = max(h1, h2) * max(h1, h2);
 
        timer.start("a, b, F calculation cycle");
        for (int i = 0; i < local_M; ++i) {
            for (int j = 0; j < local_N; ++j) {
                int gi = i_start + i;
                int gj = j_start + j;
 
                double x = get_x(gi);
                double y = get_y(gj);
 
                double x_left = x - 0.5 * h1;
                double x_right = x + 0.5 * h1;
                double y_down = y - 0.5 * h2;
                double y_up = y + 0.5 * h2;
 
                double y_l = calcIntersectionInterval(x_left, y_down, x_left, y_up, 'y');
                double x_l = calcIntersectionInterval(x_left, y_down, x_right, y_down, 'x');
 
                a(i, j) = y_l / h2;
                if (a(i, j) < 1.0) {
                    a(i, j) += (1.0 - a(i, j)) / eps;
                }
 
                b(i, j) = x_l / h1;
                if (b(i, j) < 1.0) {
                    b(i, j) += (1.0 - b(i, j)) / eps;
                }
 
                if (gi == 0 || gi == M || gj == 0 || gj == N) {
                    F(i, j) = 0.0;
                } else {
                    F(i, j) = calcIntersectionArea(x_left, x_right, y_down, y_up, stepsForArea) / (h1 * h2);
                }
            }
        }
        timer.stop("a, b, F calculation cycle");

        exchange_ghosts(a);
        exchange_ghosts(b);

        timer.start("D calculation cycle");
        for (int i = 0; i < local_M; ++i) {
            for (int j = 0; j < local_N; ++j) {
                // проверяем границу (глобальную)
                int gi = i_start + i;
                int gj = j_start + j;

                if (gi == 0 || gj == 0 || gi == M || gj == N) {
                    D(i, j) = 1.0;
                } else {
                    D(i, j) = (a(i + 1, j) + a(i, j)) / (h1 * h1) + 
                              (b(i, j + 1) + b(i, j)) / (h2 * h2);
                }
            }
        }
        timer.stop("D calculation cycle");
    }
 
    void applyA(LocalGridFunction& w, LocalGridFunction& result) const {
        exchange_ghosts(w);
        timer.start("applyA cycle");
        for (int i = 0; i < local_M; ++i) {
            for (int j = 0; j < local_N; ++j) {
                int gi = i_start + i;
                int gj = j_start + j;
 
                if (gi == 0 || gj == 0 || gi == M || gj == N) {
                    result(i, j) = w(i, j);
                } else {
                    result(i, j) = D(i, j) * w(i, j) -
                        a(i + 1, j) / (h1 * h1) * w(i + 1, j) -
                        a(i, j) / (h1 * h1) * w(i - 1, j) -
                        b(i, j + 1) / (h2 * h2) * w(i, j + 1) -
                        b(i, j) / (h2 * h2) * w(i, j - 1);
                }
            }
        }
        timer.stop("applyA cycle");
    }
};
 
struct Solver {
    LocalGridFunction w;
    LocalGridFunction p;
    LocalGridFunction r;
    LocalGridFunction z;
    int max_steps;
    int steps;
    double delta;
 
    Solver(int max_steps, double delta) : 
        max_steps(max_steps), delta(delta),
        w(local_M, local_N), p(local_M, local_N), r(local_M, local_N), z(local_M, local_N) {}

    void calcZ(const Coefficients& coef) {
        timer.start("calcZ cycle");
        for (int i = 0; i < local_M; ++i) {
            for (int j = 0; j < local_N; ++j) {
                z(i, j) = r(i, j) / coef.D(i, j);
            }
        }
        timer.stop("calcZ cycle");
    }
 
    void solve(Coefficients& coef) {
        LocalGridFunction Aw(local_M, local_N);
        LocalGridFunction& Ap = Aw;
 
        coef.applyA(w, Aw);
        calcScaledAdd(coef.F, Aw, -1, r);
        calcZ(coef);
        calcScaledAdd(z, p, 0, p);
        coef.applyA(p, Ap);
 
        double zr_scalar_prod = calcScalarProd(z, r);
        double zr_scalar_prod_prev = zr_scalar_prod;
        double alpha = zr_scalar_prod / calcScalarProd(Ap, p);
 
        int k;
        for (k = 1; k < max_steps; k++) {
            calcScaledAdd(w, p, alpha, w);
            calcScaledAdd(r, Ap, -alpha, r);
            double norm = sqrt(calcScalarProd(r, r));
            if (norm < delta) {
                steps = k;
                break;
            }
            calcZ(coef);
            zr_scalar_prod = calcScalarProd(z, r);
            double beta = zr_scalar_prod / zr_scalar_prod_prev;
            zr_scalar_prod_prev = zr_scalar_prod;
            calcScaledAdd(z, p, beta, p);
            coef.applyA(p, Ap);
            alpha = zr_scalar_prod / calcScalarProd(Ap, p);
            if (k == max_steps - 1) {
                steps = max_steps;
            }
        }
    }
 
    void cout_discrepancy() const {
        double local_max_disc = 0.0;
        for (int i = 0; i < local_M; ++i) {
            for (int j = 0; j < local_N; ++j) {
                local_max_disc = max(local_max_disc, fabs(r(i, j)));
            }
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
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }
    if (mpi_rank == 0) {
        printf("\n\nGrid size: M=%d, N=%d\n", M, N);
    }

    setup_2d_domain();
    timer.stop("Initialization");

    Coefficients coef(10);
    coef.calcCoefficients();
 
    Solver solver((M - 1) * (N - 1), 1e-8);
    solver.solve(coef);

    if (mpi_rank == 0) {
        cout << "Iteration number: " << solver.steps << endl;
        cout << "MPI processes: " << mpi_size << endl;
    }
    timer.stop("Total time");
    solver.cout_discrepancy();
    timer.report(mpi_rank, mpi_size, MPI_COMM_WORLD);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
