// src_cpp/bindings/pybind_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>

#include "relorbit/types.hpp"
#include "relorbit/api.hpp"
#include "relorbit/models/schwarzschild_equatorial.hpp"
#include "relorbit/models/kerr_equatorial.hpp"
#include "relorbit/models/schwarzschild_lowthrust.hpp"
#include "relorbit/models/kerr_lowthrust.hpp"
#include "relorbit/models/attitude.hpp"

namespace py = pybind11;

static std::vector<double> rk4_decay(double y0, double k, double t0, double tf, int n_steps) {
    if (n_steps <= 0) throw std::runtime_error("n_steps must be > 0");
    double dt = (tf - t0) / (double)n_steps;
    std::vector<double> y; y.reserve(n_steps + 1);
    auto f = [&](double, double yy) { return -k * yy; };
    double t = t0, yy = y0; y.push_back(yy);
    for (int i = 0; i < n_steps; ++i) {
        double k1=f(t,yy), k2=f(t+.5*dt,yy+.5*dt*k1), k3=f(t+.5*dt,yy+.5*dt*k2), k4=f(t+dt,yy+dt*k3);
        yy += (dt/6.0)*(k1+2*k2+2*k3+k4); t += dt; y.push_back(yy);
    }
    return y;
}

PYBIND11_MODULE(_engine, m) {
    m.doc() = "relorbit C++ engine (pybind11)";
    m.def("hello", []() { return std::string("relorbit C++ engine: OK"); });
    m.def("rk4_decay", &rk4_decay,
          py::arg("y0"),py::arg("k"),py::arg("t0"),py::arg("tf"),py::arg("n_steps"));

    // ── Enums base ─────────────────────────────────────────────
    py::enum_<relorbit::OrbitStatus>(m, "OrbitStatus")
        .value("BOUND",   relorbit::OrbitStatus::BOUND)
        .value("UNBOUND", relorbit::OrbitStatus::UNBOUND)
        .value("CAPTURE", relorbit::OrbitStatus::CAPTURE)
        .value("ERROR",   relorbit::OrbitStatus::ERROR);

    py::class_<relorbit::Maneuver>(m, "Maneuver")
        .def(py::init<>())
        .def_readwrite("tau",    &relorbit::Maneuver::tau)
        .def_readwrite("dv_r",   &relorbit::Maneuver::dv_r)
        .def_readwrite("dv_phi", &relorbit::Maneuver::dv_phi);

    py::class_<relorbit::SolverCfg>(m, "SolverCfg")
        .def(py::init<>())
        .def_readwrite("dt",           &relorbit::SolverCfg::dt)
        .def_readwrite("n_steps",      &relorbit::SolverCfg::n_steps)
        .def_readwrite("record_every", &relorbit::SolverCfg::record_every)
        .def_readwrite("maneuvers",    &relorbit::SolverCfg::maneuvers);

    // ── Newton ─────────────────────────────────────────────────
    py::class_<relorbit::TrajectoryNewton>(m, "TrajectoryNewton")
        .def_readonly("t",       &relorbit::TrajectoryNewton::t)
        .def_readonly("y",       &relorbit::TrajectoryNewton::y)
        .def_readonly("energy",  &relorbit::TrajectoryNewton::energy)
        .def_readonly("h",       &relorbit::TrajectoryNewton::h)
        .def_readonly("mass",    &relorbit::TrajectoryNewton::mass)
        .def_readonly("status",  &relorbit::TrajectoryNewton::status)
        .def_readonly("message", &relorbit::TrajectoryNewton::message);
    using NewtonVecFn = relorbit::TrajectoryNewton(*)(double,const std::vector<double>&,double,double,const relorbit::SolverCfg&);
    m.def("simulate_newton_rk4", (NewtonVecFn)&relorbit::simulate_newton_rk4,
          py::arg("mu"),py::arg("state0"),py::arg("t0"),py::arg("tf"),py::arg("cfg"));

    // ── Schwarzschild geodesica ────────────────────────────────
    py::class_<relorbit::TrajectorySchwarzschildEq>(m, "TrajectorySchwarzschildEq")
        .def_readonly("tau",            &relorbit::TrajectorySchwarzschildEq::tau)
        .def_readonly("r",              &relorbit::TrajectorySchwarzschildEq::r)
        .def_readonly("phi",            &relorbit::TrajectorySchwarzschildEq::phi)
        .def_readonly("tcoord",         &relorbit::TrajectorySchwarzschildEq::tcoord)
        .def_readonly("vcoord",         &relorbit::TrajectorySchwarzschildEq::vcoord)
        .def_property_readonly("t",     [](const relorbit::TrajectorySchwarzschildEq& tr){return tr.tcoord;})
        .def_property_readonly("v",     [](const relorbit::TrajectorySchwarzschildEq& tr){return tr.vcoord;})
        .def_readonly("pr",             &relorbit::TrajectorySchwarzschildEq::pr)
        .def_readonly("mass",           &relorbit::TrajectorySchwarzschildEq::mass)
        .def_readonly("epsilon",        &relorbit::TrajectorySchwarzschildEq::epsilon)
        .def_readonly("E_series",       &relorbit::TrajectorySchwarzschildEq::E_series)
        .def_readonly("L_series",       &relorbit::TrajectorySchwarzschildEq::L_series)
        .def_readonly("ut_fd",          &relorbit::TrajectorySchwarzschildEq::ut_fd)
        .def_readonly("vt_fd",          &relorbit::TrajectorySchwarzschildEq::vt_fd)
        .def_readonly("ur_fd",          &relorbit::TrajectorySchwarzschildEq::ur_fd)
        .def_readonly("uphi_fd",        &relorbit::TrajectorySchwarzschildEq::uphi_fd)
        .def_readonly("norm_u",         &relorbit::TrajectorySchwarzschildEq::norm_u)
        .def_readonly("ut_theory",      &relorbit::TrajectorySchwarzschildEq::ut_theory)
        .def_readonly("vt_theory",      &relorbit::TrajectorySchwarzschildEq::vt_theory)
        .def_readonly("ur_theory",      &relorbit::TrajectorySchwarzschildEq::ur_theory)
        .def_readonly("uphi_theory",    &relorbit::TrajectorySchwarzschildEq::uphi_theory)
        .def_readonly("norm_u_theory",  &relorbit::TrajectorySchwarzschildEq::norm_u_theory)
        .def_readonly("event_kind",     &relorbit::TrajectorySchwarzschildEq::event_kind)
        .def_readonly("event_tau",      &relorbit::TrajectorySchwarzschildEq::event_tau)
        .def_readonly("event_tcoord",   &relorbit::TrajectorySchwarzschildEq::event_tcoord)
        .def_readonly("event_vcoord",   &relorbit::TrajectorySchwarzschildEq::event_vcoord)
        .def_readonly("event_r",        &relorbit::TrajectorySchwarzschildEq::event_r)
        .def_readonly("event_phi",      &relorbit::TrajectorySchwarzschildEq::event_phi)
        .def_readonly("event_pr",       &relorbit::TrajectorySchwarzschildEq::event_pr)
        .def_readonly("status",         &relorbit::TrajectorySchwarzschildEq::status)
        .def_readonly("message",        &relorbit::TrajectorySchwarzschildEq::message)
        .def_readonly("M",              &relorbit::TrajectorySchwarzschildEq::M)
        .def_readonly("E",              &relorbit::TrajectorySchwarzschildEq::E)
        .def_readonly("L",              &relorbit::TrajectorySchwarzschildEq::L)
        .def_readonly("r0",             &relorbit::TrajectorySchwarzschildEq::r0)
        .def_readonly("phi0",           &relorbit::TrajectorySchwarzschildEq::phi0);
    using SchwFn = relorbit::TrajectorySchwarzschildEq(*)(double,double,double,double,double,double,double,double,const relorbit::SolverCfg&,double,double);
    m.def("simulate_schwarzschild_equatorial_rk4",(SchwFn)&relorbit::simulate_schwarzschild_equatorial_rk4,
        py::arg("M"),py::arg("E"),py::arg("L"),py::arg("r0"),py::arg("phi0"),py::arg("pr0"),
        py::arg("tau0"),py::arg("tauf"),py::arg("cfg"),py::arg("capture_r")=2.0,py::arg("capture_eps")=1e-12);

    // ── Kerr geodesica ─────────────────────────────────────────
    py::class_<relorbit::TrajectoryKerrEq>(m, "TrajectoryKerrEq")
        .def_readonly("tau",            &relorbit::TrajectoryKerrEq::tau)
        .def_readonly("r",              &relorbit::TrajectoryKerrEq::r)
        .def_readonly("phi",            &relorbit::TrajectoryKerrEq::phi)
        .def_readonly("tcoord",         &relorbit::TrajectoryKerrEq::tcoord)
        .def_readonly("vcoord",         &relorbit::TrajectoryKerrEq::vcoord)
        .def_property_readonly("t",     [](const relorbit::TrajectoryKerrEq& tr){return tr.tcoord;})
        .def_property_readonly("v",     [](const relorbit::TrajectoryKerrEq& tr){return tr.vcoord;})
        .def_readonly("pr",             &relorbit::TrajectoryKerrEq::pr)
        .def_readonly("mass",           &relorbit::TrajectoryKerrEq::mass)
        .def_readonly("epsilon",        &relorbit::TrajectoryKerrEq::epsilon)
        .def_readonly("E_series",       &relorbit::TrajectoryKerrEq::E_series)
        .def_readonly("L_series",       &relorbit::TrajectoryKerrEq::L_series)
        .def_readonly("ut_fd",          &relorbit::TrajectoryKerrEq::ut_fd)
        .def_readonly("vt_fd",          &relorbit::TrajectoryKerrEq::vt_fd)
        .def_readonly("ur_fd",          &relorbit::TrajectoryKerrEq::ur_fd)
        .def_readonly("uphi_fd",        &relorbit::TrajectoryKerrEq::uphi_fd)
        .def_readonly("norm_u",         &relorbit::TrajectoryKerrEq::norm_u)
        .def_readonly("ut_theory",      &relorbit::TrajectoryKerrEq::ut_theory)
        .def_readonly("vt_theory",      &relorbit::TrajectoryKerrEq::vt_theory)
        .def_readonly("ur_theory",      &relorbit::TrajectoryKerrEq::ur_theory)
        .def_readonly("uphi_theory",    &relorbit::TrajectoryKerrEq::uphi_theory)
        .def_readonly("norm_u_theory",  &relorbit::TrajectoryKerrEq::norm_u_theory)
        .def_readonly("event_kind",     &relorbit::TrajectoryKerrEq::event_kind)
        .def_readonly("event_tau",      &relorbit::TrajectoryKerrEq::event_tau)
        .def_readonly("event_tcoord",   &relorbit::TrajectoryKerrEq::event_tcoord)
        .def_readonly("event_vcoord",   &relorbit::TrajectoryKerrEq::event_vcoord)
        .def_readonly("event_r",        &relorbit::TrajectoryKerrEq::event_r)
        .def_readonly("event_phi",      &relorbit::TrajectoryKerrEq::event_phi)
        .def_readonly("event_pr",       &relorbit::TrajectoryKerrEq::event_pr)
        .def_readonly("status",         &relorbit::TrajectoryKerrEq::status)
        .def_readonly("message",        &relorbit::TrajectoryKerrEq::message)
        .def_readonly("M",              &relorbit::TrajectoryKerrEq::M)
        .def_readonly("a",              &relorbit::TrajectoryKerrEq::a)
        .def_readonly("E",              &relorbit::TrajectoryKerrEq::E)
        .def_readonly("L",              &relorbit::TrajectoryKerrEq::L)
        .def_readonly("r0",             &relorbit::TrajectoryKerrEq::r0)
        .def_readonly("phi0",           &relorbit::TrajectoryKerrEq::phi0);
    using KerrFn = relorbit::TrajectoryKerrEq(*)(double,double,double,double,double,double,double,double,double,const relorbit::SolverCfg&,double,double);
    m.def("simulate_kerr_equatorial_rk4",(KerrFn)&relorbit::simulate_kerr_equatorial_rk4,
        py::arg("M"),py::arg("a"),py::arg("E"),py::arg("L"),py::arg("r0"),py::arg("phi0"),py::arg("pr0"),
        py::arg("tau0"),py::arg("tauf"),py::arg("cfg"),py::arg("capture_r")=2.0,py::arg("capture_eps")=1e-12);

    // ── Low-Thrust: enums e ThrustCfg ─────────────────────────
    py::enum_<relorbit::ThrustMode>(m, "ThrustMode")
        .value("CONSTANT",        relorbit::ThrustMode::CONSTANT)
        .value("TANGENTIAL_ONLY", relorbit::ThrustMode::TANGENTIAL_ONLY)
        .value("RADIAL_ONLY",     relorbit::ThrustMode::RADIAL_ONLY)
        .value("COAST",           relorbit::ThrustMode::COAST);

    py::class_<relorbit::ThrustCfg>(m, "ThrustCfg")
        .def(py::init<>())
        .def_readwrite("F_r",          &relorbit::ThrustCfg::F_r)
        .def_readwrite("F_phi",        &relorbit::ThrustCfg::F_phi)
        .def_readwrite("isp_s",        &relorbit::ThrustCfg::isp_s)
        .def_readwrite("mass0_kg",     &relorbit::ThrustCfg::mass0_kg)
        .def_readwrite("dry_mass_kg",  &relorbit::ThrustCfg::dry_mass_kg)
        .def_readwrite("mode",         &relorbit::ThrustCfg::mode)
        .def_readwrite("tau_on",       &relorbit::ThrustCfg::tau_on)
        .def_readwrite("tau_off",      &relorbit::ThrustCfg::tau_off);

    // ── Schwarzschild Low-Thrust ───────────────────────────────
    py::class_<relorbit::TrajectorySchwarzschildLT>(m, "TrajectorySchwarzschildLT")
        .def_readonly("tau",        &relorbit::TrajectorySchwarzschildLT::tau)
        .def_readonly("r",          &relorbit::TrajectorySchwarzschildLT::r)
        .def_readonly("phi",        &relorbit::TrajectorySchwarzschildLT::phi)
        .def_readonly("pr",         &relorbit::TrajectorySchwarzschildLT::pr)
        .def_readonly("L",          &relorbit::TrajectorySchwarzschildLT::L)
        .def_readonly("E",          &relorbit::TrajectorySchwarzschildLT::E)
        .def_readonly("mass",       &relorbit::TrajectorySchwarzschildLT::mass)
        .def_readonly("epsilon",    &relorbit::TrajectorySchwarzschildLT::epsilon)
        .def_readonly("thrust_mag", &relorbit::TrajectorySchwarzschildLT::thrust_mag)
        .def_readonly("event_kind", &relorbit::TrajectorySchwarzschildLT::event_kind)
        .def_readonly("event_tau",  &relorbit::TrajectorySchwarzschildLT::event_tau)
        .def_readonly("event_r",    &relorbit::TrajectorySchwarzschildLT::event_r)
        .def_readonly("event_phi",  &relorbit::TrajectorySchwarzschildLT::event_phi)
        .def_readonly("event_pr",   &relorbit::TrajectorySchwarzschildLT::event_pr)
        .def_readonly("event_L",    &relorbit::TrajectorySchwarzschildLT::event_L)
        .def_readonly("event_E",    &relorbit::TrajectorySchwarzschildLT::event_E)
        .def_readonly("event_mass", &relorbit::TrajectorySchwarzschildLT::event_mass)
        .def_readonly("status",     &relorbit::TrajectorySchwarzschildLT::status)
        .def_readonly("message",    &relorbit::TrajectorySchwarzschildLT::message)
        .def_readonly("E0",         &relorbit::TrajectorySchwarzschildLT::E0)
        .def_readonly("L0",         &relorbit::TrajectorySchwarzschildLT::L0);
    m.def("simulate_schwarzschild_lowthrust_rk4",
          &relorbit::simulate_schwarzschild_lowthrust_rk4,
          py::arg("M"),py::arg("E0"),py::arg("L0"),py::arg("r0"),py::arg("phi0"),py::arg("pr0"),
          py::arg("tau0"),py::arg("tauf"),py::arg("thrust"),py::arg("cfg"),
          py::arg("capture_r")=2.0,py::arg("capture_eps")=1e-12);

    // ── Kerr Low-Thrust ────────────────────────────────────────
    py::class_<relorbit::TrajectoryKerrLT>(m, "TrajectoryKerrLT")
        .def_readonly("tau",        &relorbit::TrajectoryKerrLT::tau)
        .def_readonly("r",          &relorbit::TrajectoryKerrLT::r)
        .def_readonly("phi",        &relorbit::TrajectoryKerrLT::phi)
        .def_readonly("pr",         &relorbit::TrajectoryKerrLT::pr)
        .def_readonly("L",          &relorbit::TrajectoryKerrLT::L)
        .def_readonly("E",          &relorbit::TrajectoryKerrLT::E)
        .def_readonly("mass",       &relorbit::TrajectoryKerrLT::mass)
        .def_readonly("epsilon",    &relorbit::TrajectoryKerrLT::epsilon)
        .def_readonly("thrust_mag", &relorbit::TrajectoryKerrLT::thrust_mag)
        .def_readonly("event_kind", &relorbit::TrajectoryKerrLT::event_kind)
        .def_readonly("event_tau",  &relorbit::TrajectoryKerrLT::event_tau)
        .def_readonly("event_r",    &relorbit::TrajectoryKerrLT::event_r)
        .def_readonly("event_phi",  &relorbit::TrajectoryKerrLT::event_phi)
        .def_readonly("event_pr",   &relorbit::TrajectoryKerrLT::event_pr)
        .def_readonly("event_L",    &relorbit::TrajectoryKerrLT::event_L)
        .def_readonly("event_E",    &relorbit::TrajectoryKerrLT::event_E)
        .def_readonly("event_mass", &relorbit::TrajectoryKerrLT::event_mass)
        .def_readonly("status",     &relorbit::TrajectoryKerrLT::status)
        .def_readonly("message",    &relorbit::TrajectoryKerrLT::message)
        .def_readonly("E0",         &relorbit::TrajectoryKerrLT::E0)
        .def_readonly("L0",         &relorbit::TrajectoryKerrLT::L0);
    m.def("simulate_kerr_lowthrust_rk4",
          &relorbit::simulate_kerr_lowthrust_rk4,
          py::arg("M"),py::arg("a"),py::arg("E0"),py::arg("L0"),py::arg("r0"),py::arg("phi0"),py::arg("pr0"),
          py::arg("tau0"),py::arg("tauf"),py::arg("thrust"),py::arg("cfg"),
          py::arg("capture_r")=2.0,py::arg("capture_eps")=1e-12);

    // ══════════════════════════════════════════════════════════
    // ── Atitude 6-DOF com Quaternions (Item 7) ────────────────
    // ══════════════════════════════════════════════════════════

    // ── AttitudeCfg ───────────────────────────────────────────
    py::class_<relorbit::AttitudeCfg>(m, "AttitudeCfg")
        .def(py::init<>())
        .def_readwrite("dt",           &relorbit::AttitudeCfg::dt)
        .def_readwrite("n_steps",      &relorbit::AttitudeCfg::n_steps)
        .def_readwrite("record_every", &relorbit::AttitudeCfg::record_every)
        .def_readwrite("renorm_every", &relorbit::AttitudeCfg::renorm_every)
        .def_readwrite("renorm_tol",   &relorbit::AttitudeCfg::renorm_tol);

    // ── TorqueCfg ─────────────────────────────────────────────
    py::class_<relorbit::TorqueCfg>(m, "TorqueCfg")
        .def(py::init<>())
        .def_readwrite("tx",    &relorbit::TorqueCfg::tx)
        .def_readwrite("ty",    &relorbit::TorqueCfg::ty)
        .def_readwrite("tz",    &relorbit::TorqueCfg::tz)
        .def_readwrite("t_on",  &relorbit::TorqueCfg::t_on)
        .def_readwrite("t_off", &relorbit::TorqueCfg::t_off)
        .def("active",  &relorbit::TorqueCfg::active,  py::arg("t"))
        .def("get_x",   &relorbit::TorqueCfg::get_x,   py::arg("t"))
        .def("get_y",   &relorbit::TorqueCfg::get_y,   py::arg("t"))
        .def("get_z",   &relorbit::TorqueCfg::get_z,   py::arg("t"));

    // ── InertiaTensor ─────────────────────────────────────────
    py::class_<relorbit::InertiaTensor>(m, "InertiaTensor")
        .def(py::init<>())
        .def_readwrite("I", &relorbit::InertiaTensor::I)
        .def_static("diagonal",
            &relorbit::InertiaTensor::diagonal,
            py::arg("Ixx"), py::arg("Iyy"), py::arg("Izz"))
        .def_static("full",
            &relorbit::InertiaTensor::full,
            py::arg("Ixx"), py::arg("Iyy"), py::arg("Izz"),
            py::arg("Ixy"), py::arg("Ixz"), py::arg("Iyz"))
        .def("T_rot",
            &relorbit::InertiaTensor::T_rot,
            py::arg("wx"), py::arg("wy"), py::arg("wz"))
        .def("mul",
            [](const relorbit::InertiaTensor& it, double vx, double vy, double vz){
                return it.mul(vx, vy, vz);
            },
            py::arg("vx"), py::arg("vy"), py::arg("vz"));

    // ── AttitudeState ─────────────────────────────────────────
    py::class_<relorbit::AttitudeState>(m, "AttitudeState")
        .def(py::init<>())
        .def_readwrite("q0", &relorbit::AttitudeState::q0)
        .def_readwrite("q1", &relorbit::AttitudeState::q1)
        .def_readwrite("q2", &relorbit::AttitudeState::q2)
        .def_readwrite("q3", &relorbit::AttitudeState::q3)
        .def_readwrite("wx", &relorbit::AttitudeState::wx)
        .def_readwrite("wy", &relorbit::AttitudeState::wy)
        .def_readwrite("wz", &relorbit::AttitudeState::wz)
        .def("qnorm",        &relorbit::AttitudeState::qnorm)
        .def("renormalize",  &relorbit::AttitudeState::renormalize);

    // ── TrajectoryAttitude ────────────────────────────────────
    py::class_<relorbit::TrajectoryAttitude>(m, "TrajectoryAttitude")
        .def_readonly("t",            &relorbit::TrajectoryAttitude::t)
        .def_readonly("q0",           &relorbit::TrajectoryAttitude::q0)
        .def_readonly("q1",           &relorbit::TrajectoryAttitude::q1)
        .def_readonly("q2",           &relorbit::TrajectoryAttitude::q2)
        .def_readonly("q3",           &relorbit::TrajectoryAttitude::q3)
        .def_readonly("wx",           &relorbit::TrajectoryAttitude::wx)
        .def_readonly("wy",           &relorbit::TrajectoryAttitude::wy)
        .def_readonly("wz",           &relorbit::TrajectoryAttitude::wz)
        .def_readonly("qnorm",        &relorbit::TrajectoryAttitude::qnorm)
        .def_readonly("T_rot",        &relorbit::TrajectoryAttitude::T_rot)
        .def_readonly("renorm_delta", &relorbit::TrajectoryAttitude::renorm_delta)
        .def_readonly("Ixx",          &relorbit::TrajectoryAttitude::Ixx)
        .def_readonly("Iyy",          &relorbit::TrajectoryAttitude::Iyy)
        .def_readonly("Izz",          &relorbit::TrajectoryAttitude::Izz)
        .def_readonly("Ixy",          &relorbit::TrajectoryAttitude::Ixy)
        .def_readonly("Ixz",          &relorbit::TrajectoryAttitude::Ixz)
        .def_readonly("Iyz",          &relorbit::TrajectoryAttitude::Iyz)
        .def_readonly("status",       &relorbit::TrajectoryAttitude::status)
        .def_readonly("message",      &relorbit::TrajectoryAttitude::message);

    // ── dcm_from_quaternion ───────────────────────────────────
    m.def("dcm_from_quaternion",
        [](double q0, double q1, double q2, double q3) {
            auto R = relorbit::dcm_from_quaternion(q0, q1, q2, q3);
            return std::vector<double>(R.begin(), R.end());
        },
        py::arg("q0"), py::arg("q1"), py::arg("q2"), py::arg("q3"),
        "DCM R ∈ SO(3) body→inercial a partir do quaternion. "
        "Devolve lista de 9 floats row-major.");

    // ── simulate_attitude_rk4 ─────────────────────────────────
    m.def("simulate_attitude_rk4",
        &relorbit::simulate_attitude_rk4,
        py::arg("state0"),
        py::arg("inertia"),
        py::arg("torque"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("cfg"),
        "Integra dinâmica de atitude 6-DOF (q + ω) com RK4 de passo fixo. "
        "Critérios: ‖q‖=1 (renormalização controlada) e T_rot conservada sem torque.");

} // PYBIND11_MODULE