#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>

using namespace std;
 
// (M + 1) и (N + 1) узлов по оси x и y соответственно
// x - индекс i, y - индекс j
int M = 40, N = 40;
double h1, h2;

class TimeLogger {
private:
    std::unordered_map<std::string, double> start_times;
    std::unordered_map<std::string, double> accumulated_times;

    double now() {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
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

    void report() const {
        cout << "\n====== TIME REPORT ======\n";
        double total_cycle_time = 0.0;
        for (const auto& kv: accumulated_times) {
            const string& label = kv.first;
            double local_time = kv.second;
            cout << " - " << label << ": " << local_time << "s\n";
        }
        cout << "=========================\n\n";
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
 
class GridFunction {
public:
    double* data;
    int M, N;
 
    GridFunction(int M, int N) : M(M), N(N) {
        data = new double[(M+1)*(N+1)];
        for(int i=0; i<=(M+1)*(N+1)-1;i++) data[i]=0.0;
    }

    ~GridFunction() {
        delete[] data;
    }
 
    double& operator()(int i, int j) {
        return data[i * (N + 1) + j];
    }
 
    const double& operator()(int i, int j) const {
        return data[i * (N + 1) + j];
    }
};

 
double calcScalarProd(const GridFunction& u, const GridFunction& v) {
    timer.start("calcScalarProd cycle");
    double sum = 0.0;
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            sum += u(i, j) * v(i, j) * h1 * h2;
        }
    }
    timer.stop("calcScalarProd cycle");
    return sum;
}
 
void calcScaledAdd(const GridFunction& A, const GridFunction& B, double alpha, GridFunction& result) {
    timer.start("calcScaledAdd cycle");
    if (alpha == 0.0) {
        const bool resultIsA = (&A == &result);
        if (!resultIsA) {
            for (int i = 0; i <= M; ++i) {
                for (int j = 0; j <= N; ++j) {
                    result(i, j) = A(i, j);
                }
            }
        }
    } else {
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                result(i, j) = A(i, j) + alpha * B(i, j);
            }
        }
    }
    timer.stop("calcScaledAdd cycle");
}
 
struct Coefficients {
    GridFunction a;
    GridFunction b;
    GridFunction F;
    GridFunction D;
    int stepsForArea;
 
    Coefficients(int steps) : stepsForArea(steps), a(M, N), b(M, N),
        F(M, N), D(M, N){}
 
    void calcCoefficients() {
        double eps = max(h1, h2) * max(h1, h2);
 
        timer.start("a, b, F calculation cycle");
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                double x = get_x(i);
                double y = get_y(j);
 
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
 
                if (i == 0 || i == M || j == 0 || j == N) {
                    F(i, j) = 0.0;
                } else {
                    F(i, j) = calcIntersectionArea(x_left, x_right, y_down, y_up, stepsForArea) / (h1 * h2);
                }
            }
        }
        timer.stop("a, b, F calculation cycle");

        timer.start("D calculation cycle");
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                if (i == 0 || j == 0 || i == M || j == N) {
                    D(i, j) = 1.0;
                } else {
                    D(i, j) = (a(i + 1, j) + a(i, j)) / (h1 * h1) + 
                              (b(i, j + 1) + b(i, j)) / (h2 * h2);
                }
            }
        }
        timer.stop("D calculation cycle");
    }
 
    void applyA(GridFunction& w, GridFunction& result) const {
        timer.start("applyA cycle");
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                if (i == 0 || j == 0 || i == M || j == N) {
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
    GridFunction w;
    GridFunction p;
    GridFunction r;
    GridFunction z;
    int max_steps;
    int steps;
    double delta;
    double divide_eps;
 
    Solver(int max_steps, double delta, double divide_eps = 1e-10) : 
        max_steps(max_steps), delta(delta), divide_eps(divide_eps),
        w(M, N), p(M, N), r(M, N), z(M, N) {}

    void calcZ(const Coefficients& coef) {
        timer.start("calcZ cycle");
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                z(i, j) = r(i, j) / (coef.D(i, j) + divide_eps);
            }
        }
        timer.stop("calcZ cycle");
    }
 
    void solve(Coefficients& coef) {
        GridFunction Aw(M, N);
        GridFunction& Ap = Aw;
 
        coef.applyA(w, Aw);
        calcScaledAdd(coef.F, Aw, -1, r);
        calcZ(coef);
        calcScaledAdd(z, p, 0, p);
        coef.applyA(p, Ap);
 
        double zr_scalar_prod = calcScalarProd(z, r);
        double zr_scalar_prod_prev = zr_scalar_prod;
        double alpha = zr_scalar_prod / (calcScalarProd(Ap, p) + divide_eps);
 
        int k;
        for (k = 1; k < max_steps; k++) {
            calcScaledAdd(w, p, alpha, w);
            calcScaledAdd(r, Ap, -alpha, r);
            double norm = fabs(alpha) * sqrt(calcScalarProd(p, p));
            if (norm < delta) {
                steps = k;
                break;
            }
            calcZ(coef);
            zr_scalar_prod = calcScalarProd(z, r);
            double beta = zr_scalar_prod / (zr_scalar_prod_prev + divide_eps);
            zr_scalar_prod_prev = zr_scalar_prod;
            calcScaledAdd(z, p, beta, p);
            coef.applyA(p, Ap);
            alpha = zr_scalar_prod / (calcScalarProd(Ap, p) + divide_eps);
            if (k == max_steps - 1) {
                steps = max_steps;
            }
        }
    }
 
    void cout_discrepancy() const {
        double max_disc = 0.0;
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                max_disc = max(max_disc, fabs(r(i, j)));
            }
        }
        cout << "Solution discrepancy: " << max_disc << endl;
    }
};
 
int main(int argc, char** argv) {
    timer.start("Total time");
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }
    printf("\n\nGrid size: M=%d, N=%d\n", M, N);

    h1 = 4.0 / M;
    h2 = 3.0 / N;

    Coefficients coef(10);
    coef.calcCoefficients();

    Solver solver((M - 1) * (N - 1), 1e-20, 1e-40);
    solver.solve(coef);

    timer.stop("Total time");
    timer.report();

    cout << "Iteration number: " << solver.steps << endl;
    solver.cout_discrepancy();
    return 0;
}
