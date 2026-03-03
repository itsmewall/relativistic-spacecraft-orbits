// src_cpp/include/relorbit/solvers/hermite.hpp
#pragma once
#include <cmath>
#include <algorithm>

namespace relorbit {

// Avalia o polinômio cúbico de Hermite no intervalo alpha em [0, 1]
inline double hermite_eval(double x0, double x1, double dx0, double dx1, double h, double alpha) {
    double v0 = dx0 * h;
    double v1 = dx1 * h;
    double a2 = alpha * alpha;
    double a3 = a2 * alpha;
    return (2.0 * a3 - 3.0 * a2 + 1.0) * x0 + 
           (a3 - 2.0 * a2 + alpha) * v0 + 
           (-2.0 * a3 + 3.0 * a2) * x1 + 
           (a3 - a2) * v1;
}

// Acha a raiz de H(alpha) = x_target via método de Newton-Raphson
inline double hermite_root(double x0, double x1, double dx0, double dx1, double h, double x_target) {
    // Chute inicial linear (lerp clássico)
    double alpha = (x1 != x0) ? (x_target - x0) / (x1 - x0) : 0.5;
    alpha = std::clamp(alpha, 0.0, 1.0);
    
    double v0 = dx0 * h;
    double v1 = dx1 * h;
    
    // 5 a 7 iterações costumam bater no limite da máquina (1e-15 precisão)
    for (int i = 0; i < 7; ++i) {
        double a2 = alpha * alpha;
        double a3 = a2 * alpha;
        
        // Valor de H(alpha)
        double val = (2.0 * a3 - 3.0 * a2 + 1.0) * x0 + 
                     (a3 - 2.0 * a2 + alpha) * v0 + 
                     (-2.0 * a3 + 3.0 * a2) * x1 + 
                     (a3 - a2) * v1;
                     
        // Derivada H'(alpha)
        double der = (6.0 * a2 - 6.0 * alpha) * x0 + 
                     (3.0 * a2 - 4.0 * alpha + 1.0) * v0 + 
                     (-6.0 * a2 + 6.0 * alpha) * x1 + 
                     (3.0 * a2 - 2.0 * alpha) * v1;
        
        double diff = val - x_target;
        if (std::abs(diff) < 1e-14 || std::abs(der) < 1e-14) break;
        
        alpha -= diff / der;
        alpha = std::clamp(alpha, 0.0, 1.0);
    }
    return alpha;
}

} // namespace relorbit