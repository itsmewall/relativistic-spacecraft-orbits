// src_cpp/include/relorbit/models/kerr_equatorial.hpp
#pragma once
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "relorbit/types.hpp"

namespace relorbit {

struct TrajectoryKerrEq {
    std::vector<double> tau;
    std::vector<double> r;
    std::vector<double> phi;
    std::vector<double> tcoord;
    std::vector<double> vcoord; 
    std::vector<double> pr;

    std::vector<double> epsilon; 
    std::vector<double> E_series;
    std::vector<double> L_series;

    std::vector<double> ut_fd;
    std::vector<double> vt_fd;
    std::vector<double> ur_fd;
    std::vector<double> uphi_fd;
    std::vector<double> norm_u;

    std::vector<double> ut_theory;
    std::vector<double> vt_theory;
    std::vector<double> ur_theory;
    std::vector<double> uphi_theory;
    std::vector<double> norm_u_theory;

    std::vector<std::string> event_kind;
    std::vector<double> event_tau;
    std::vector<double> event_tcoord;
    std::vector<double> event_vcoord;
    std::vector<double> event_r;
    std::vector<double> event_phi;
    std::vector<double> event_pr;

    OrbitStatus status = OrbitStatus::ERROR;
    std::string message;

    double M = 0.0;
    double a = 0.0;
    double E = 0.0;
    double L = 0.0;
    double r0 = 0.0;
    double phi0 = 0.0;
};

TrajectoryKerrEq simulate_kerr_equatorial_rk4(
    double M, double a, double E, double L,
    double r0, double phi0, double pr0,
    double tau0, double tauf,
    const SolverCfg& cfg,
    double capture_r = 2.0,
    double capture_eps = 1e-12
);

} // namespace relorbit