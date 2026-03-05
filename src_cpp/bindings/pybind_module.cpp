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
#include "relorbit/models/schwarzschild_6dof.hpp"
#include "relorbit/models/kerr_6dof.hpp"
#include "relorbit/gr/kerr_null_geodesic.hpp"

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
        .def_property("tx",   &relorbit::TorqueCfg::tx,  &relorbit::TorqueCfg::set_tx)
        .def_property("ty",   &relorbit::TorqueCfg::ty,  &relorbit::TorqueCfg::set_ty)
        .def_property("tz",   &relorbit::TorqueCfg::tz,  &relorbit::TorqueCfg::set_tz)
        .def_readwrite("t_on",  &relorbit::TorqueCfg::t_on)
        .def_readwrite("t_off", &relorbit::TorqueCfg::t_off)
        .def("active", &relorbit::TorqueCfg::active, py::arg("t"))
        .def("get_x",  &relorbit::TorqueCfg::get_x,  py::arg("t"))
        .def("get_y",  &relorbit::TorqueCfg::get_y,  py::arg("t"))
        .def("get_z",  &relorbit::TorqueCfg::get_z,  py::arg("t"));

    // ── InertiaTensor ─────────────────────────────────────────
    py::class_<relorbit::InertiaTensor>(m, "InertiaTensor")
        .def(py::init<>())
        .def_property("I",
            [](const relorbit::InertiaTensor& it) {
                std::vector<double> v(9);
                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(v.data()) = it.I;
                return v;
            },
            [](relorbit::InertiaTensor& it, const std::vector<double>& v) {
                if (v.size() != 9) throw std::invalid_argument("I must have 9 elements");
                it.I = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(v.data());
            })
        .def_static("diagonal",
            &relorbit::InertiaTensor::diagonal,
            py::arg("Ixx"), py::arg("Iyy"), py::arg("Izz"))
        .def_static("full",
            &relorbit::InertiaTensor::full,
            py::arg("Ixx"), py::arg("Iyy"), py::arg("Izz"),
            py::arg("Ixy"), py::arg("Ixz"), py::arg("Iyz"))
        .def("T_rot",
            [](const relorbit::InertiaTensor& it, double wx, double wy, double wz) {
                return it.T_rot(relorbit::Vec3(wx, wy, wz));
            },
            py::arg("wx"), py::arg("wy"), py::arg("wz"))
        .def("mul",
            [](const relorbit::InertiaTensor& it, double vx, double vy, double vz) {
                relorbit::Vec3 r = it.mul(relorbit::Vec3(vx, vy, vz));
                return std::array<double,3>{r[0], r[1], r[2]};
            },
            py::arg("vx"), py::arg("vy"), py::arg("vz"));

    // ── AttitudeState ─────────────────────────────────────────
    py::class_<relorbit::AttitudeState>(m, "AttitudeState")
        .def(py::init<>())
        .def_property("q0", &relorbit::AttitudeState::q0, &relorbit::AttitudeState::set_q0)
        .def_property("q1", &relorbit::AttitudeState::q1, &relorbit::AttitudeState::set_q1)
        .def_property("q2", &relorbit::AttitudeState::q2, &relorbit::AttitudeState::set_q2)
        .def_property("q3", &relorbit::AttitudeState::q3, &relorbit::AttitudeState::set_q3)
        .def_property("wx", &relorbit::AttitudeState::wx, &relorbit::AttitudeState::set_wx)
        .def_property("wy", &relorbit::AttitudeState::wy, &relorbit::AttitudeState::set_wy)
        .def_property("wz", &relorbit::AttitudeState::wz, &relorbit::AttitudeState::set_wz)
        .def("qnorm",       &relorbit::AttitudeState::qnorm)
        .def("renormalize", &relorbit::AttitudeState::renormalize);

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
            relorbit::Mat3 R = relorbit::dcm_from_quaternion(q0, q1, q2, q3);
            std::vector<double> v(9);
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(v.data()) = R;
            return v;
        },
        py::arg("q0"), py::arg("q1"), py::arg("q2"), py::arg("q3"),
        "DCM R in SO(3) body->inercial a partir do quaternion. "
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
        "Integra dinamica de atitude 6-DOF (q + w) com RK4 de passo fixo. "
        "Criterios: ||q||=1 (renormalizacao controlada) e T_rot conservada sem torque.");

    // ══════════════════════════════════════════════════════════════
    // Item 8 — Thrust Vectoring 6-DOF (Schwarzschild acoplado)
    // ══════════════════════════════════════════════════════════════

    // ── EngineCfg ─────────────────────────────────────────────────
    py::class_<relorbit::EngineCfg>(m, "EngineCfg")
        .def(py::init<>())
        .def_readwrite("F_newton",        &relorbit::EngineCfg::F_newton)
        // ── PATCH Item8 ──────────────────────────────────────────
        // a_geom_override > 0: substitui completamente F/(m*c²) por este valor
        // fixo em unidades geométricas [M⁻¹].
        //
        // Raiz do problema: F_geom = F/(m*c²) usa c_SI=3×10⁸ m/s, mas o
        // integrador Kerr opera em coords geométricas onde c=G=1.  O factor
        // c² = 9×10¹⁶ torna a_geom ≈ 3×10⁻¹⁹ M⁻¹ para F=30 N, ou seja
        // 9×10¹⁶ vezes menor do que o necessário para produzir ΔL > 0.05.
        //
        // Fix sem alterar a física existente: se a_geom_override > 0, usa-o
        // directamente. Para validação do Item 8 usa:
        //   a_geom_override = F_newton / mass0_kg   (c=1, natural)
        //
        // active() também dispara quando a_geom_override > 0 (mesmo F_newton=0).
        .def_readwrite("a_geom_override", &relorbit::EngineCfg::a_geom_override)
        // ─────────────────────────────────────────────────────────
        .def_readwrite("isp_s",           &relorbit::EngineCfg::isp_s)
        .def_readwrite("tau_on",          &relorbit::EngineCfg::tau_on)
        .def_readwrite("tau_off",         &relorbit::EngineCfg::tau_off)
        .def_readwrite("mass0_kg",        &relorbit::EngineCfg::mass0_kg)
        .def_readwrite("dry_mass_kg",     &relorbit::EngineCfg::dry_mass_kg)
        .def_property("nozzle_body",
            [](const relorbit::EngineCfg& e) {
                return std::array<double,3>{e.nozzle_body[0], e.nozzle_body[1], e.nozzle_body[2]};
            },
            [](relorbit::EngineCfg& e, const std::array<double,3>& v) {
                e.nozzle_body = relorbit::Vec3(v[0], v[1], v[2]);
            })
        .def_property("torque_reaction",
            [](const relorbit::EngineCfg& e) {
                return std::array<double,3>{e.torque_reaction[0], e.torque_reaction[1], e.torque_reaction[2]};
            },
            [](relorbit::EngineCfg& e, const std::array<double,3>& v) {
                e.torque_reaction = relorbit::Vec3(v[0], v[1], v[2]);
            })
        .def("active",  &relorbit::EngineCfg::active,  py::arg("tau"))
        .def("F_geom",  &relorbit::EngineCfg::F_geom,  py::arg("m_kg"));

    // ── AttitudeCfg6DOF ───────────────────────────────────────────
    py::class_<relorbit::AttitudeCfg6DOF>(m, "AttitudeCfg6DOF")
        .def(py::init<>())
        .def_readwrite("inertia",       &relorbit::AttitudeCfg6DOF::inertia)
        .def_readwrite("ext_torque",    &relorbit::AttitudeCfg6DOF::ext_torque)
        .def_readwrite("renorm_every",  &relorbit::AttitudeCfg6DOF::renorm_every)
        .def_readwrite("renorm_tol",    &relorbit::AttitudeCfg6DOF::renorm_tol);

    // ── SolverCfg6DOF ─────────────────────────────────────────────
    py::class_<relorbit::SolverCfg6DOF>(m, "SolverCfg6DOF")
        .def(py::init<>())
        .def_readwrite("dt",            &relorbit::SolverCfg6DOF::dt)
        .def_readwrite("n_steps",       &relorbit::SolverCfg6DOF::n_steps)
        .def_readwrite("record_every",  &relorbit::SolverCfg6DOF::record_every)
        .def_readwrite("renorm_every",  &relorbit::SolverCfg6DOF::renorm_every)
        .def_readwrite("renorm_tol",    &relorbit::SolverCfg6DOF::renorm_tol)
        .def_readwrite("capture_r",     &relorbit::SolverCfg6DOF::capture_r)
        .def_readwrite("capture_eps",   &relorbit::SolverCfg6DOF::capture_eps);

    // ── TrajectoryCoupled ─────────────────────────────────────────
    py::class_<relorbit::TrajectoryCoupled>(m, "TrajectoryCoupled")
        // Órbita
        .def_readonly("tau",          &relorbit::TrajectoryCoupled::tau)
        .def_readonly("r",            &relorbit::TrajectoryCoupled::r)
        .def_readonly("phi",          &relorbit::TrajectoryCoupled::phi)
        .def_readonly("pr",           &relorbit::TrajectoryCoupled::pr)
        .def_readonly("E",            &relorbit::TrajectoryCoupled::E)
        .def_readonly("L",            &relorbit::TrajectoryCoupled::L)
        .def_readonly("mass",         &relorbit::TrajectoryCoupled::mass)
        .def_readonly("epsilon",      &relorbit::TrajectoryCoupled::epsilon)
        .def_readonly("tcoord",       &relorbit::TrajectoryCoupled::tcoord)
        // Atitude
        .def_readonly("q0",           &relorbit::TrajectoryCoupled::q0)
        .def_readonly("q1",           &relorbit::TrajectoryCoupled::q1)
        .def_readonly("q2",           &relorbit::TrajectoryCoupled::q2)
        .def_readonly("q3",           &relorbit::TrajectoryCoupled::q3)
        .def_readonly("wx",           &relorbit::TrajectoryCoupled::wx)
        .def_readonly("wy",           &relorbit::TrajectoryCoupled::wy)
        .def_readonly("wz",           &relorbit::TrajectoryCoupled::wz)
        .def_readonly("qnorm",        &relorbit::TrajectoryCoupled::qnorm)
        .def_readonly("T_rot",        &relorbit::TrajectoryCoupled::T_rot)
        // Thrust
        .def_readonly("thrust_r",     &relorbit::TrajectoryCoupled::thrust_r)
        .def_readonly("thrust_phi",   &relorbit::TrajectoryCoupled::thrust_phi)
        .def_readonly("pointing_err", &relorbit::TrajectoryCoupled::pointing_err)
        // Meta
        .def_readonly("M",            &relorbit::TrajectoryCoupled::M)
        .def_property_readonly("status", [](const relorbit::TrajectoryCoupled& t) {
            switch (t.status) {
                case relorbit::OrbitStatus::BOUND:   return std::string("BOUND");
                case relorbit::OrbitStatus::UNBOUND: return std::string("UNBOUND");
                case relorbit::OrbitStatus::CAPTURE: return std::string("CAPTURE");
                default:                             return std::string("ERROR");
            }
        })
        .def_readonly("message",      &relorbit::TrajectoryCoupled::message);

    // ── simulate_schwarzschild_6dof_rk4 ──────────────────────────
    m.def("simulate_schwarzschild_6dof_rk4",
        &relorbit::simulate_schwarzschild_6dof_rk4,
        py::arg("M"),
        py::arg("E0"), py::arg("L0"),
        py::arg("r0"), py::arg("phi0"), py::arg("pr0"),
        py::arg("att0"),
        py::arg("tau0"), py::arg("tauf"),
        py::arg("engine"),
        py::arg("att_cfg"),
        py::arg("cfg"),
        "Integra orbita + atitude acopladas em Schwarzschild equatorial. "
        "O empuxo e vectorizado pela orientacao actual do quaternion: "
        "a_thrust = (F/m) * R(q) * nozzle_body.");

    // ── TidalModel enum ───────────────────────────────────────────
    py::enum_<relorbit::TidalModel>(m, "TidalModel")
        .value("NONE",       relorbit::TidalModel::NONE)
        .value("WEAK_N",     relorbit::TidalModel::WEAK_N)
        .value("DIAG_EIJ",   relorbit::TidalModel::DIAG_EIJ)
        .value("RIEMANN_FD", relorbit::TidalModel::RIEMANN_FD);

    // ── TidalCfg ──────────────────────────────────────────────────
    py::class_<relorbit::TidalCfg>(m, "TidalCfg")
        .def(py::init<>())
        .def_readwrite("enabled",         &relorbit::TidalCfg::enabled)
        .def_readwrite("model",           &relorbit::TidalCfg::model)
        .def_readwrite("fd_eps_r",        &relorbit::TidalCfg::fd_eps_r)
        .def_readwrite("Q_from_inertia",  &relorbit::TidalCfg::Q_from_inertia)
        .def_readwrite("spin_correction", &relorbit::TidalCfg::spin_correction);

    // ── AttitudeCfgKerr ───────────────────────────────────────────
    py::class_<relorbit::AttitudeCfgKerr>(m, "AttitudeCfgKerr")
        .def(py::init<>())
        .def_readwrite("inertia",       &relorbit::AttitudeCfgKerr::inertia)
        .def_readwrite("ext_torque",    &relorbit::AttitudeCfgKerr::ext_torque)
        .def_readwrite("renorm_every",  &relorbit::AttitudeCfgKerr::renorm_every)
        .def_readwrite("renorm_tol",    &relorbit::AttitudeCfgKerr::renorm_tol)
        .def_readwrite("tidal",         &relorbit::AttitudeCfgKerr::tidal);

    // ── TrajectoryCoupledKerr ─────────────────────────────────────
    py::class_<relorbit::TrajectoryCoupledKerr>(m, "TrajectoryCoupledKerr")
        // Órbita
        .def_readonly("tau",          &relorbit::TrajectoryCoupledKerr::tau)
        .def_readonly("r",            &relorbit::TrajectoryCoupledKerr::r)
        .def_readonly("phi",          &relorbit::TrajectoryCoupledKerr::phi)
        .def_readonly("pr",           &relorbit::TrajectoryCoupledKerr::pr)
        .def_readonly("E",            &relorbit::TrajectoryCoupledKerr::E)
        .def_readonly("L",            &relorbit::TrajectoryCoupledKerr::L)
        .def_readonly("mass",         &relorbit::TrajectoryCoupledKerr::mass)
        .def_readonly("epsilon",      &relorbit::TrajectoryCoupledKerr::epsilon)
        .def_readonly("tcoord",       &relorbit::TrajectoryCoupledKerr::tcoord)
        // Atitude
        .def_readonly("q0",           &relorbit::TrajectoryCoupledKerr::q0)
        .def_readonly("q1",           &relorbit::TrajectoryCoupledKerr::q1)
        .def_readonly("q2",           &relorbit::TrajectoryCoupledKerr::q2)
        .def_readonly("q3",           &relorbit::TrajectoryCoupledKerr::q3)
        .def_readonly("wx",           &relorbit::TrajectoryCoupledKerr::wx)
        .def_readonly("wy",           &relorbit::TrajectoryCoupledKerr::wy)
        .def_readonly("wz",           &relorbit::TrajectoryCoupledKerr::wz)
        .def_readonly("qnorm",        &relorbit::TrajectoryCoupledKerr::qnorm)
        .def_readonly("T_rot",        &relorbit::TrajectoryCoupledKerr::T_rot)
        // Empuxo
        .def_readonly("thrust_r",     &relorbit::TrajectoryCoupledKerr::thrust_r)
        .def_readonly("thrust_phi",   &relorbit::TrajectoryCoupledKerr::thrust_phi)
        .def_readonly("pointing_err", &relorbit::TrajectoryCoupledKerr::pointing_err)
        // Maré
        .def_readonly("tidal_tau_x",  &relorbit::TrajectoryCoupledKerr::tidal_tau_x)
        .def_readonly("tidal_tau_y",  &relorbit::TrajectoryCoupledKerr::tidal_tau_y)
        .def_readonly("tidal_tau_z",  &relorbit::TrajectoryCoupledKerr::tidal_tau_z)
        .def_readonly("tidal_norm",   &relorbit::TrajectoryCoupledKerr::tidal_norm)
        .def_readonly("align_angle_rad", &relorbit::TrajectoryCoupledKerr::align_angle_rad)
        .def_readonly("tidal_E_norm", &relorbit::TrajectoryCoupledKerr::tidal_E_norm)
        // Meta
        .def_readonly("M",            &relorbit::TrajectoryCoupledKerr::M)
        .def_readonly("a",            &relorbit::TrajectoryCoupledKerr::a)
        .def_property_readonly("status", [](const relorbit::TrajectoryCoupledKerr& t) {
            switch (t.status) {
                case relorbit::OrbitStatus::BOUND:   return std::string("BOUND");
                case relorbit::OrbitStatus::UNBOUND: return std::string("UNBOUND");
                case relorbit::OrbitStatus::CAPTURE: return std::string("CAPTURE");
                default:                             return std::string("ERROR");
            }
        })
        .def_readonly("message",      &relorbit::TrajectoryCoupledKerr::message);

    // ── simulate_kerr_6dof_rk4 ───────────────────────────────────
    m.def("simulate_kerr_6dof_rk4",
        &relorbit::simulate_kerr_6dof_rk4,
        py::arg("M"), py::arg("a"),
        py::arg("E0"), py::arg("L0"),
        py::arg("r0"), py::arg("phi0"), py::arg("pr0"),
        py::arg("att0"),
        py::arg("tau0"), py::arg("tauf"),
        py::arg("engine"),
        py::arg("att_cfg"),
        py::arg("cfg"),
        "Integra orbita + atitude + torque de mare acoplados em Kerr equatorial. "
        "Modelos de mare: WEAK_N, DIAG_EIJ, RIEMANN_FD (modo monstro).");

    // ══════════════════════════════════════════════════════════════
    // ── Geodésicas Nulas Kerr — Ray Tracing de Telemetria ─────────
    // ══════════════════════════════════════════════════════════════

    // ── NullGeodesicConfig ────────────────────────────────────────
    py::class_<relorbit::gr::NullGeodesicConfig>(m, "NullGeodesicConfig")
        .def(py::init<>())
        .def_readwrite("M",          &relorbit::gr::NullGeodesicConfig::M)
        .def_readwrite("a",          &relorbit::gr::NullGeodesicConfig::a)
        .def_readwrite("r_obs",      &relorbit::gr::NullGeodesicConfig::r_obs)
        .def_readwrite("n_lut",      &relorbit::gr::NullGeodesicConfig::n_lut)
        .def_readwrite("n_steps",    &relorbit::gr::NullGeodesicConfig::n_steps)
        .def_readwrite("dl_coarse",  &relorbit::gr::NullGeodesicConfig::dl_coarse)
        .def_readwrite("dl_fine",    &relorbit::gr::NullGeodesicConfig::dl_fine)
        .def_readwrite("r_switch",   &relorbit::gr::NullGeodesicConfig::r_switch)
        .def_readwrite("n_bisect",   &relorbit::gr::NullGeodesicConfig::n_bisect)
        .def("r_horizon",            &relorbit::gr::NullGeodesicConfig::r_horizon)
        .def("b_crit_approx",        &relorbit::gr::NullGeodesicConfig::b_crit_approx);

    // ── NullRayResult ─────────────────────────────────────────────
    py::class_<relorbit::gr::NullRayResult>(m, "NullRayResult")
        .def_readonly("b",          &relorbit::gr::NullRayResult::b)
        .def_readonly("dphi",       &relorbit::gr::NullRayResult::dphi)
        .def_readonly("t_coord",    &relorbit::gr::NullRayResult::t_coord)
        .def_readonly("captured",   &relorbit::gr::NullRayResult::captured)
        .def_readonly("n_turns",    &relorbit::gr::NullRayResult::n_turns);

    // ── NullGeodesicLUT ───────────────────────────────────────────
    py::class_<relorbit::gr::NullGeodesicLUT>(m, "NullGeodesicLUT")
        .def_readonly("b_arr",      &relorbit::gr::NullGeodesicLUT::b_arr)
        .def_readonly("phi_arr",    &relorbit::gr::NullGeodesicLUT::phi_arr)
        .def_readonly("t_arr",      &relorbit::gr::NullGeodesicLUT::t_arr)
        .def_readonly("cap_arr",    &relorbit::gr::NullGeodesicLUT::cap_arr)
        .def_readonly("wind_arr",   &relorbit::gr::NullGeodesicLUT::wind_arr)
        .def_readonly("r_s",        &relorbit::gr::NullGeodesicLUT::r_s)
        .def("n_arrived",           &relorbit::gr::NullGeodesicLUT::n_arrived)
        .def("n_captured",          &relorbit::gr::NullGeodesicLUT::n_captured)
        .def("query_phi",           &relorbit::gr::NullGeodesicLUT::query_phi,
             py::arg("dphi_target"), py::arg("winding") = 0,
             "Interpola b* para um dado dphi_target. "
             "Retorna (b_star, t_star) ou (nan, nan) se fora do intervalo.");

    // ── TelemetrySignal ───────────────────────────────────────────
    py::class_<relorbit::gr::TelemetrySignal::Image>(m, "TelemetryImage")
        .def_readonly("b",           &relorbit::gr::TelemetrySignal::Image::b)
        .def_readonly("dphi",        &relorbit::gr::TelemetrySignal::Image::dphi)
        .def_readonly("t_coord",     &relorbit::gr::TelemetrySignal::Image::t_coord)
        .def_readonly("redshift_z",  &relorbit::gr::TelemetrySignal::Image::redshift_z)
        .def_readonly("time_delay",  &relorbit::gr::TelemetrySignal::Image::time_delay);

    py::class_<relorbit::gr::TelemetrySignal>(m, "TelemetrySignal")
        .def_readonly("tau_s",       &relorbit::gr::TelemetrySignal::tau_s)
        .def_readonly("r_s",         &relorbit::gr::TelemetrySignal::r_s)
        .def_readonly("phi_s",       &relorbit::gr::TelemetrySignal::phi_s)
        .def_readonly("visible",     &relorbit::gr::TelemetrySignal::visible)
        .def_readonly("n_images",    &relorbit::gr::TelemetrySignal::n_images)
        .def_readonly("images",      &relorbit::gr::TelemetrySignal::images);

    // ── RayTracerOptions ──────────────────────────────────────────
    py::class_<relorbit::gr::RayTracerOptions>(m, "RayTracerOptions")
        .def(py::init<>())
        .def_readwrite("receiver_r",        &relorbit::gr::RayTracerOptions::receiver_r)
        .def_readwrite("receiver_phi",      &relorbit::gr::RayTracerOptions::receiver_phi)
        .def_readwrite("n_images_max",      &relorbit::gr::RayTracerOptions::n_images_max)
        .def_readwrite("compute_redshift",  &relorbit::gr::RayTracerOptions::compute_redshift)
        .def_readwrite("compute_delay",     &relorbit::gr::RayTracerOptions::compute_delay);

    // ── Funções livres ────────────────────────────────────────────
    m.def("integrate_null_ray",
        &relorbit::gr::integrate_null_ray,
        py::arg("cfg"), py::arg("b"), py::arg("r_s"), py::arg("dl") = 0.5,
        "Integra uma geodésica nula Kerr equatorial de r_s até r_obs (ou captura). "
        "Retorna NullRayResult com delta_phi, t_coord e status.");

    m.def("build_null_geodesic_lut",
        &relorbit::gr::build_null_geodesic_lut,
        py::arg("cfg"), py::arg("r_s"),
        "Constrói LUT de N_lut raios nulos para r_s fixo. "
        "Cada query subsequente custa O(log N) em vez de O(N_steps).");

    m.def("bisect_impact_parameter",
        [](const relorbit::gr::NullGeodesicConfig& cfg,
           double r_s, double phi_s, double phi_obs,
           int winding,
           const relorbit::gr::NullGeodesicLUT* lut)
           -> py::object
        {
            auto res = relorbit::gr::bisect_impact_parameter(
                cfg, r_s, phi_s, phi_obs, winding, lut);
            if (!res.has_value()) return py::none();
            return py::cast(*res);
        },
        py::arg("cfg"), py::arg("r_s"), py::arg("phi_s"), py::arg("phi_obs"),
        py::arg("winding") = 0, py::arg("lut") = nullptr,
        "Encontra b* tal que delta_phi(b*) = phi_obs - phi_s (+ winding). "
        "Usa LUT como bracket inicial se fornecida. Retorna NullRayResult ou None.");

    m.def("compute_redshift_kerr",
        &relorbit::gr::compute_redshift_kerr,
        py::arg("M"), py::arg("a"), py::arg("b"),
        py::arg("r_s"), py::arg("r_obs"), py::arg("omega_s") = 0.0,
        "Factor 1+z combinado (gravitacional + Doppler) para geodésica com parâmetro b. "
        "omega_s = velocidade angular da nave [rad/M]; 0 = receptor estático.");

    m.def("circular_orbit_omega",
        &relorbit::gr::circular_orbit_omega,
        py::arg("M"), py::arg("a"), py::arg("r"), py::arg("prograde") = true,
        "Velocidade angular da órbita circular Kerr: sqrt(M) / (r^1.5 +/- a*sqrt(M)).");

    m.def("raytrace_trajectory",
        &relorbit::gr::raytrace_trajectory,
        py::arg("cfg"), py::arg("opts"),
        py::arg("tau_arr"), py::arg("r_arr"), py::arg("phi_arr"), py::arg("omega_arr"),
        "Ray tracer completo para arrays de posição (tau, r, phi). "
        "Constrói LUT uma vez para r_s mediano e consulta por busca binária. "
        "Retorna lista de TelemetrySignal com visibilidade, b*, redshift e atraso Shapiro.");
}