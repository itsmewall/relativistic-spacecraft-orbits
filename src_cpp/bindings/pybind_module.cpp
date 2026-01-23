// src_cpp/pybind/_engine.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <stdexcept>

#include "relorbit/types.hpp"
#include "relorbit/api.hpp"
#include "relorbit/models/schwarzschild_equatorial.hpp"

namespace py = pybind11;

// toy antigo (sanity)
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

    m.def("hello", []() { return std::string("relorbit C++ engine: OK"); });

    m.def("rk4_decay", &rk4_decay,
          py::arg("y0"), py::arg("k"), py::arg("t0"), py::arg("tf"), py::arg("n_steps"),
          "Integrate dy/dt=-k*y with RK4 fixed-step. Returns y(t) samples.");

    py::enum_<relorbit::OrbitStatus>(m, "OrbitStatus")
        .value("BOUND", relorbit::OrbitStatus::BOUND)
        .value("UNBOUND", relorbit::OrbitStatus::UNBOUND)
        .value("CAPTURE", relorbit::OrbitStatus::CAPTURE)
        .value("ERROR", relorbit::OrbitStatus::ERROR);

    py::class_<relorbit::SolverCfg>(m, "SolverCfg")
        .def(py::init<>())
        .def_readwrite("dt", &relorbit::SolverCfg::dt)
        .def_readwrite("n_steps", &relorbit::SolverCfg::n_steps);

    py::class_<relorbit::TrajectoryNewton>(m, "TrajectoryNewton")
        .def_readonly("t", &relorbit::TrajectoryNewton::t)
        .def_readonly("y", &relorbit::TrajectoryNewton::y)
        .def_readonly("energy", &relorbit::TrajectoryNewton::energy)
        .def_readonly("h", &relorbit::TrajectoryNewton::h)
        .def_readonly("status", &relorbit::TrajectoryNewton::status)
        .def_readonly("message", &relorbit::TrajectoryNewton::message);

    // ponteiro explícito para o overload vector
    using NewtonVecFn = relorbit::TrajectoryNewton (*)(
        double, const std::vector<double>&, double, double, const relorbit::SolverCfg&
    );
    NewtonVecFn newton_vec = &relorbit::simulate_newton_rk4;

    m.def("simulate_newton_rk4", newton_vec,
          py::arg("mu"),
          py::arg("state0"),
          py::arg("t0"),
          py::arg("tf"),
          py::arg("cfg"),
          "Simulate Newtonian 2-body planar orbit with RK4 fixed-step (state0=[x,y,vx,vy]).");

    py::class_<relorbit::TrajectorySchwarzschildEq>(m, "TrajectorySchwarzschildEq")
        .def_readonly("tau", &relorbit::TrajectorySchwarzschildEq::tau)
        .def_readonly("r", &relorbit::TrajectorySchwarzschildEq::r)
        .def_readonly("phi", &relorbit::TrajectorySchwarzschildEq::phi)

        // t(τ) — coordenada temporal Schwarzschild (singular no horizonte)
        .def_readonly("tcoord", &relorbit::TrajectorySchwarzschildEq::tcoord)

        // v(τ) — tempo regular no horizonte (ingoing Eddington–Finkelstein)
        .def_readonly("vcoord", &relorbit::TrajectorySchwarzschildEq::vcoord)

        // Alias opcional p/ ergonomia em Python
        .def_property_readonly("t", [](const relorbit::TrajectorySchwarzschildEq& tr) {
            return tr.tcoord;
        })
        .def_property_readonly("v", [](const relorbit::TrajectorySchwarzschildEq& tr) {
            return tr.vcoord;
        })

        .def_readonly("pr", &relorbit::TrajectorySchwarzschildEq::pr)
        .def_readonly("epsilon", &relorbit::TrajectorySchwarzschildEq::epsilon)
        .def_readonly("E_series", &relorbit::TrajectorySchwarzschildEq::E_series)
        .def_readonly("L_series", &relorbit::TrajectorySchwarzschildEq::L_series)

        // FD (finite-difference)
        .def_readonly("ut_fd", &relorbit::TrajectorySchwarzschildEq::ut_fd)
        .def_readonly("vt_fd", &relorbit::TrajectorySchwarzschildEq::vt_fd)
        .def_readonly("ur_fd", &relorbit::TrajectorySchwarzschildEq::ur_fd)
        .def_readonly("uphi_fd", &relorbit::TrajectorySchwarzschildEq::uphi_fd)
        .def_readonly("norm_u", &relorbit::TrajectorySchwarzschildEq::norm_u)

        // séries por construção (theory)
        .def_readonly("ut_theory", &relorbit::TrajectorySchwarzschildEq::ut_theory)
        .def_readonly("vt_theory", &relorbit::TrajectorySchwarzschildEq::vt_theory)
        .def_readonly("ur_theory", &relorbit::TrajectorySchwarzschildEq::ur_theory)
        .def_readonly("uphi_theory", &relorbit::TrajectorySchwarzschildEq::uphi_theory)
        .def_readonly("norm_u_theory", &relorbit::TrajectorySchwarzschildEq::norm_u_theory)

        // eventos (para validate.py)
        .def_readonly("event_kind", &relorbit::TrajectorySchwarzschildEq::event_kind)
        .def_readonly("event_tau", &relorbit::TrajectorySchwarzschildEq::event_tau)
        .def_readonly("event_tcoord", &relorbit::TrajectorySchwarzschildEq::event_tcoord)
        .def_readonly("event_vcoord", &relorbit::TrajectorySchwarzschildEq::event_vcoord)
        .def_readonly("event_r", &relorbit::TrajectorySchwarzschildEq::event_r)
        .def_readonly("event_phi", &relorbit::TrajectorySchwarzschildEq::event_phi)
        .def_readonly("event_pr", &relorbit::TrajectorySchwarzschildEq::event_pr)

        .def_readonly("status", &relorbit::TrajectorySchwarzschildEq::status)
        .def_readonly("message", &relorbit::TrajectorySchwarzschildEq::message)
        .def_readonly("M", &relorbit::TrajectorySchwarzschildEq::M)
        .def_readonly("E", &relorbit::TrajectorySchwarzschildEq::E)
        .def_readonly("L", &relorbit::TrajectorySchwarzschildEq::L)
        .def_readonly("r0", &relorbit::TrajectorySchwarzschildEq::r0)
        .def_readonly("phi0", &relorbit::TrajectorySchwarzschildEq::phi0);

    // --- Schw: ponteiro explícito para fixar a assinatura (com pr0) ---
    using SchwFn = relorbit::TrajectorySchwarzschildEq (*)(
        double, double, double, double, double, double, double, double,
        const relorbit::SolverCfg&, double, double
    );
    SchwFn schw_fn = &relorbit::simulate_schwarzschild_equatorial_rk4;

    m.def(
        "simulate_schwarzschild_equatorial_rk4",
        schw_fn,
        py::arg("M"),
        py::arg("E"),
        py::arg("L"),
        py::arg("r0"),
        py::arg("phi0"),
        py::arg("pr0"),
        py::arg("tau0"),
        py::arg("tauf"),
        py::arg("cfg"),
        py::arg("capture_r") = 2.0,
        py::arg("capture_eps") = 1e-12
    );
}
