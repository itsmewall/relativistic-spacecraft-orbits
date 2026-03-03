# src/relorbit_py/validate.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from . import engine_hello
from .simulate import load_cases_yaml
from .validate_helpers import fmt_e, fmt_f, short_msg
from .validate_models import (
    validate_newton,
    validate_schw,
    validate_kerr,
    run_convergence_newton_one_case,
    schw_signature,
    kerr_signature,
    check_convergence_schw,
    check_convergence_events_schw,
)
from .validate_perihelion import (
    validate_schw_perihelion,
    validate_schw_isco,
)

# ============================================================
# Pretty output (terminal)
# ============================================================

def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

def _print_table(rows: List[List[str]], headers: List[str]) -> None:
    if not rows:
        return
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))

    def fmt_row(r: List[str]) -> str:
        return "  ".join(r[i].ljust(widths[i]) for i in range(cols))

    print(fmt_row(headers))
    print("  ".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        print(fmt_row(r))

# ============================================================
# Suite Handlers (Refatoração Limpa)
# ============================================================

def handle_newton_suite(cases: List[Dict[str, Any]], args: Any, plotdir: str) -> Dict[str, Any]:
    results = []
    ok_cases = True

    for c in cases:
        r = validate_newton(c, plotdir if args.plots else None)
        results.append(r)
        ok_cases = ok_cases and bool(r["passed"])

    _print_header(f"Newton suite: ok={ok_cases} cases={len(cases)}")
    rows = []
    for r in results:
        rows.append([
            "PASS" if r.get("passed") else "FAIL",
            str(r.get("name")),
            f"{float(r.get('dt',0)):.1e}",
            fmt_e(r.get("energy_rel_drift"), width=12),
            fmt_e(r.get("h_rel_drift"), width=12),
            str(r.get("status", "")),
            str(r.get("status_theory", "") or ""),
        ])
    _print_table(rows, headers=["ok", "case", "dt", "dE_rel", "dh_rel", "status", "theory"])

    block: Dict[str, Any] = {
        "suite": "newton",
        "ok": bool(ok_cases),
        "n_cases": len(cases),
        "results": results,
    }

    if args.convergence:
        conv_reports = []
        conv_ok = True
        for c in cases:
            cr = run_convergence_newton_one_case(
                c,
                plotdir if (args.plots or args.convergence) else None,
                rigorous=bool(args.conv_rigorous),
            )
            conv_reports.append(cr)
            conv_ok = conv_ok and bool(cr.get("passed", False))

        _print_header(f"Newton convergence: ok={conv_ok} groups={len(conv_reports)}")
        c_rows = []
        for g in conv_reports:
            tag = "PASS" if g.get("passed") else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            dts = g.get("dt_effective", [])
            dts_str = ", ".join([f"{float(dt):.2e}" for dt in dts]) if dts else "-"
            c_rows.append([
                tag,
                str(g.get("name")),
                dts_str,
                fmt_e(g.get("e_dt"), width=12),
                fmt_e(g.get("e_dt2"), width=12),
                fmt_f(g.get("p_obs"), width=8, prec=3),
                fmt_e(g.get("abs_err_proxy"), width=12),
                fmt_e(g.get("rel_err_proxy"), width=12),
                short_msg(str(g.get("reason", ""))),
            ])
        _print_table(c_rows, headers=["ok", "case", "dt_eff (dt,dt/2,dt/4)", "e_dt", "e_dt2", "p_obs", "abs_err", "rel_err", "reason"])

        block["ok_convergence"] = bool(conv_ok)
        block["convergence"] = conv_reports
        block["ok_total"] = bool(ok_cases and conv_ok)

    return block


def handle_gr_suite(suite_key: str, title: str, cases: List[Dict[str, Any]], args: Any, plotdir: str, time_plotdir: str, validator: Any, sig_func: Any) -> Dict[str, Any]:
    results = []
    ok_cases = True

    for c in cases:
        rr = validator(c, plotdir if args.plots else None, time_plotdir if args.plots else None)
        if sig_func:
            rr["_sig"] = sig_func(c)
        results.append(rr)
        ok_cases = ok_cases and bool(rr.get("passed"))

    conv = check_convergence_schw(results, abs_tol=1e-9, rel_tol=0.25)
    conv_ok = all(bool(x.get("passed")) for x in conv) if conv else True

    events_conv = check_convergence_events_schw(results, abs_tol_factor=2.0, rel_tol=0.0)
    events_conv_ok = all(bool(x.get("passed")) for x in events_conv) if events_conv else True

    ok_total = bool(ok_cases and conv_ok and events_conv_ok)

    _print_header(f"{title} suite: ok={ok_total} (cases_ok={ok_cases}, conv_ok={conv_ok}, events_conv_ok={events_conv_ok}) cases={len(cases)}")

    rows = []
    for r in results:
        rows.append([
            "PASS" if r.get("passed") else "FAIL",
            str(r.get("name")),
            f"{float(r.get('dt',0)):.1e}",
            fmt_f(r.get("r_min"), width=10, prec=6),
            fmt_f(r.get("r_end"), width=10, prec=6),
            fmt_e(r.get("constraint_abs_max"), width=12),
            fmt_e(r.get("norm_u_abs_max"), width=12),
            str(r.get("status", "")),
            str(r.get("events_compact", "") or ""),
            short_msg(str(r.get("message", ""))),
        ])
    _print_table(rows, headers=["ok", "case", "dt", "r_min", "r_end", "eps_max", "norm_u", "status", "events", "msg"])

    # Time-dilation checks
    _print_header(f"{title} time-dilation checks (t(τ), v(τ), dt/dτ, dv/dτ)")
    td_rows = []
    for r in results:
        if not r.get("tcoord_present") and not r.get("vcoord_present"):
            continue
        td_rows.append([
            "OK" if r.get("passed") else "WARN/FAIL",
            str(r.get("name")),
            "yes" if r.get("tcoord_present") else "no",
            "yes" if r.get("tcoord_finite_ok") else "no",
            "yes" if r.get("tcoord_monotone_ok") else "no",
            fmt_e(r.get("dt_dtau_rel_max"), width=12),
            fmt_e(r.get("dt_dtau_abs_max"), width=12),
            "yes" if r.get("vcoord_present") else "no",
            "yes" if r.get("vcoord_finite_ok") else "no",
            "yes" if r.get("vcoord_monotone_ok") else "no",
            fmt_e(r.get("dv_dtau_rel_max"), width=12),
            fmt_e(r.get("dv_dtau_abs_max"), width=12),
            str(r.get("time_mask_n", "")),
        ])
    if td_rows:
        _print_table(td_rows, headers=["ok", "case", "t", "t_finite", "t_mono", "dt_rel", "dt_abs", "v", "v_finite", "v_mono", "dv_rel", "dv_abs", "mask_n"])
    else:
        print("No time-dilation data available in this suite.")

    if conv:
        _print_header(f"{title} convergence: norm_u_abs_max should not increase when dt decreases")
        c_rows2 = []
        for g in conv:
            tag = "PASS" if g.get("passed") else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            dts = ", ".join([f"{float(dt):.2e}" for dt in g.get("dts", [])])
            nus = ", ".join(["None" if v is None else f"{float(v):.3e}" for v in g.get("norm_u_abs_max", [])])
            c_rows2.append([tag, dts, nus, ", ".join(g.get("cases", []))])
        _print_table(c_rows2, headers=["ok", "dt (big->small)", "norm_u_abs_max", "cases"])
        for g in conv:
            for v in g.get("violations", []):
                print(f"violation: dt {float(v['dt_big']):.2e}->{float(v['dt_small']):.2e} norm_u {float(v['nu_big']):.3e}->{float(v['nu_small']):.3e} (abs_tol={float(v['abs_tol']):.1e}, rel_tol={float(v['rel_tol']):.2f})")

    if events_conv:
        _print_header(f"{title} convergence: event times should change little when dt decreases")
        e_rows = []
        for g in events_conv:
            tag = "PASS" if g.get("passed") else ("SKIP" if g.get("skipped") else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL"))
            dts = ", ".join([f"{float(dt):.2e}" for dt in g.get("dts", [])])
            reason = str(g.get("reason", "")) if (g.get("skipped") or g.get("inconclusive")) else ""
            e_rows.append([tag, dts, ", ".join(g.get("cases", [])), reason])
        _print_table(e_rows, headers=["ok", "dt (big->small)", "cases", "reason"])
        for g in events_conv:
            for mm in g.get("mismatches", []):
                print(f"mismatch: {mm.get('kind','?')} count dt {float(mm.get('dt_big',0.0)):.2e}->{float(mm.get('dt_small',0.0)):.2e} {mm.get('count_big','?')}->{mm.get('count_small','?')}")
            for v in g.get("violations", []):
                print(f"violation: {v['kind']}[{v['occurrence']}] dt {float(v['dt_big']):.2e}->{float(v['dt_small']):.2e} tau {float(v['tau_big']):.6g}->{float(v['tau_small']):.6g} abs_err={float(v['abs_err']):.3e} allowed={float(v['allowed']):.3e}")

    return {
        "suite": suite_key,
        "ok": ok_total,
        "ok_cases": ok_cases,
        "ok_convergence": conv_ok,
        "ok_events_convergence": events_conv_ok,
        "n_cases": len(cases),
        "results": results,
        "convergence": conv,
        "events_convergence": events_conv,
    }


def handle_precession_suite(cases: List[Dict[str, Any]], args: Any, plotdir: str, time_plotdir: str) -> Dict[str, Any]:
    results = []
    ok_cases = True

    for c in cases:
        rr = validate_schw_perihelion(c, plotdir if args.plots else None, time_plotdir if args.plots else None)
        rr["_sig"] = schw_signature(c)
        results.append(rr)
        ok_cases = ok_cases and bool(rr.get("passed"))

    _print_header(f"Perihelion precession suite: ok={ok_cases} cases={len(cases)}")
    p_rows = []
    for r in results:
        used_exact = r.get("precession_used_exact", False)
        p_rows.append([
            "PASS" if r.get("passed") else "FAIL",
            str(r.get("name")),
            f"{float(r.get('dt',0)):.1e}",
            str(int(r.get("precession_n_orbits", 0))),
            fmt_e(r.get("precession_delta_phi_mean"), width=10),
            fmt_e(r.get("precession_delta_phi_exact"), width=10),
            fmt_e(r.get("precession_delta_phi_pn"), width=10),
            "exact" if used_exact else "PN",
            fmt_e(r.get("precession_rel_err"), width=10),
            fmt_e(r.get("precession_consistency"), width=8),
            short_msg(str(r.get("message", ""))),
        ])
    _print_table(p_rows, headers=["ok", "case", "dt", "n_orb", "Δφ_sim", "Δφ_exact", "Δφ_PN", "th", "rel_err", "consist", "msg"])

    _print_header("Perihelion precession — expected values (reference)")
    print(f"{'case':<42} {'Δφ_PN (rad)':>14} {'Δφ_PN (arcsec/orb)':>20}")
    for r in results:
        dphi_pn_val = float(r.get("precession_delta_phi_pn", 0.0) or 0.0)
        arcsec = dphi_pn_val * (180.0 * 3600.0 / 3.14159265358979)
        dphi_th_val = r.get("precession_delta_phi_theory")
        th_str = f"{dphi_th_val:.6f}" if dphi_th_val is not None else "N/A"
        print(f"  {r.get('name', ''):<40} Δφ_PN={dphi_pn_val:.6f}  ({arcsec:,.1f}\")  Δφ_exact={th_str}")

    return {
        "suite": "perihelion_precession",
        "ok": bool(ok_cases),
        "n_cases": len(cases),
        "results": results,
    }


def handle_isco_suite(cases: List[Dict[str, Any]], args: Any, plotdir: str, time_plotdir: str) -> Dict[str, Any]:
    results = []
    ok_cases = True

    for c in cases:
        rr = validate_schw_isco(c, plotdir if args.plots else None, time_plotdir if args.plots else None)
        results.append(rr)
        ok_cases = ok_cases and bool(rr.get("passed"))

    _print_header(f"ISCO stability suite: ok={ok_cases} cases={len(cases)}")
    i_rows = []
    for r in results:
        i_rows.append([
            "PASS" if r.get("passed") else "FAIL",
            str(r.get("name")),
            f"{float(r.get('dt',0)):.1e}",
            fmt_f(r.get("isco_r_start"), width=8, prec=4),
            fmt_f(r.get("isco_r_M"), width=8, prec=4),
            "stable" if r.get("isco_theory_stable") else "unstable",
            str(r.get("status", "")),
            "ok" if r.get("isco_ok") else "FAIL",
            short_msg(str(r.get("message", ""))),
        ])
    _print_table(i_rows, headers=["ok", "case", "dt", "r_start", "r_isco", "theory", "status", "isco_check", "msg"])

    print("\n  r_ISCO = 6M  (in geometric units G=c=1)")
    print("  r > 6M → stable orbit (BOUND expected)")
    print("  r < 6M → unstable orbit (CAPTURE expected)")

    return {
        "suite": "isco",
        "ok": bool(ok_cases),
        "n_cases": len(cases),
        "results": results,
    }


# ============================================================
# Main Loop (Polimórfico e Mínimo)
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "cases.yaml"))
    ap.add_argument("--plots", action="store_true", help="Generate plots in <out>/plots and time plots in <out>/time_plots")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--convergence", action="store_true", help="Run automatic dt refinement (dt, dt/2, dt/4) for Newton cases and estimate observed order.")
    ap.add_argument("--conv-rigorous", action="store_true", help="Use stricter defaults for convergence criteria.")

    args = ap.parse_args()

    print(engine_hello())

    cfg = load_cases_yaml(args.cases)
    outdir = args.out

    plotdir = os.path.join(outdir, "plots")
    time_plotdir = os.path.join(outdir, "time_plots")

    if args.plots or args.convergence:
        os.makedirs(plotdir, exist_ok=True)
    if args.plots:
        os.makedirs(time_plotdir, exist_ok=True)

    report: Dict[str, Any] = {"suites": []}
    suites = cfg.get("suites", {})

    for suite_key, suite_data in suites.items():
        cases = suite_data.get("cases", [])
        if not cases:
            continue

        if suite_key == "newton":
            rep = handle_newton_suite(cases, args, plotdir)
        elif suite_key == "perihelion_precession":
            rep = handle_precession_suite(cases, args, plotdir, time_plotdir)
        elif suite_key == "isco":
            rep = handle_isco_suite(cases, args, plotdir, time_plotdir)
        elif suite_key in ("schwarzschild", "kerr_equatorial", "kerr"):
            validator = validate_kerr if "kerr" in suite_key else validate_schw
            sig_func = kerr_signature if "kerr" in suite_key else schw_signature
            title = "Kerr Equatorial" if "kerr" in suite_key else "Schwarzschild"
            rep = handle_gr_suite(suite_key, title, cases, args, plotdir, time_plotdir, validator, sig_func)
        else:
            print(f"Warning: Unknown suite handler for '{suite_key}', skipping.")
            continue

        report["suites"].append(rep)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.plots or args.convergence:
        print(f"\nPlots em: {plotdir}")
    if args.plots:
        print(f"Time plots em: {time_plotdir}")
    print(f"Relatório em: {os.path.join(outdir, 'report.json')}")


if __name__ == "__main__":
    main()