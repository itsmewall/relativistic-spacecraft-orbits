#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <string>
#include <vector>

#include "relorbit/api.hpp"

namespace py = pybind11;

// Mantém o teste antigo
static std::vector<double> rk4_decay(double y0, double k, double t0, double tf, int n_steps) {
    if (n_steps <= 0) throw std::runtime_error("n_steps must be > 0");
    double dt = (tf - t0) / static_cast<double>(n_steps);
    std::vector<double> y;
    y.reserve(static_cast<size_t>(n_steps) + 1);

    auto f = [&](double /*t*/, double yy) { return -k * yy; };

    double t = t0;
    double yy = y0;
    y.push_back(yy);

    for (int i = 0; i < n_steps; ++i) {
        double k1 = f(t, yy);
        double k2 = f(t + 0.5 * dt, yy + 0.5 * dt * k1);
        double k3 = f(t + 0.5 * dt, yy + 0.5 * dt * k2);
        double k4 = f(t + dt, yy + dt * k3);
        yy = yy + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
        y.push_back(yy);
    }
    return y;
}

PYBIND11_MODULE(_engine, m) {
    m.doc() = "relorbit C++ engine (pybind11)";

    m.def("hello", []() {
        return std::string("relorbit C++ engine: OK");
    });

    m.def("rk4_decay", &rk4_decay,
          py::arg("y0"), py::arg("k"), py::arg("t0"), py::arg("tf"), py::arg("n_steps"),
          "Integrate dy/dt=-k*y with RK4 fixed-step. Returns y(t) samples.");

    // Expor SolverCfg
    py::class_<relorbit::SolverCfg>(m, "SolverCfg")
        .def(py::init<>())
        .def_readwrite("dt", &relorbit::SolverCfg::dt)
        .def_readwrite("n_steps", &relorbit::SolverCfg::n_steps);

    // Expor TrajectoryNewton
    py::class_<relorbit::TrajectoryNewton>(m, "TrajectoryNewton")
        .def_readonly("t", &relorbit::TrajectoryNewton::t)
        .def_readonly("y", &relorbit::TrajectoryNewton::y)
        .def_readonly("energy", &relorbit::TrajectoryNewton::energy)
        .def_readonly("h", &relorbit::TrajectoryNewton::h)
        .def_readonly("status", &relorbit::TrajectoryNewton::status)
        .def_readonly("message", &relorbit::TrajectoryNewton::message);

    // Expor OrbitStatus como enum
    py::enum_<relorbit::OrbitStatus>(m, "OrbitStatus")
        .value("BOUND", relorbit::OrbitStatus::BOUND)
        .value("UNBOUND", relorbit::OrbitStatus::UNBOUND)
        .value("CAPTURE", relorbit::OrbitStatus::CAPTURE)
        .value("ERROR", relorbit::OrbitStatus::ERROR);

    // Expor simulate_newton_rk4
    m.def("simulate_newton_rk4", &relorbit::simulate_newton_rk4,
          py::arg("mu"),
          py::arg("state0"),
          py::arg("t0"),
          py::arg("tf"),
          py::arg("cfg"),
          "Simulate Newtonian 2-body planar orbit with RK4 fixed-step.");
}
