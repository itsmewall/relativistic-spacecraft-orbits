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
    theoretical_precession_schw,
    _pn_precession,
)


# ============================================================
# Pretty output (terminal)
# ============================================================

def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _print_table(rows: List[List[str]], headers: List[str]) -> None:
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
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "cases.yaml"))
    ap.add_argument("--plots", action="store_true",
                    help="Generate plots in <out>/plots and time plots in <out>/time_plots")
    ap.add_argument("--out", default="out", help="Output directory")

    ap.add_argument(
        "--convergence",
        action="store_true",
        help="Run automatic dt refinement (dt, dt/2, dt/4) for Newton cases and estimate observed order.",
    )
    ap.add_argument(
        "--conv-rigorous",
        action="store_true",
        help="Use stricter defaults for convergence criteria.",
    )

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

    # ----------------------------
    # Newton
    # ----------------------------
    if "newton" in suites:
        newton_cases = suites["newton"]["cases"]
        newton_results: List[Dict[str, Any]] = []
        ok_newton = True

        for c in newton_cases:
            r = validate_newton(c, plotdir if args.plots else None)
            newton_results.append(r)
            ok_newton = ok_newton and bool(r["passed"])

        _print_header(f"Newton suite: ok={ok_newton} cases={len(newton_cases)}")
        n_rows: List[List[str]] = []
        for r in newton_results:
            n_rows.append([
                "PASS" if r["passed"] else "FAIL",
                str(r["name"]),
                f"{float(r['dt']):.1e}",
                fmt_e(r["energy_rel_drift"], width=12),
                fmt_e(r["h_rel_drift"], width=12),
                str(r.get("status", "")),
                str(r.get("status_theory", "")) if r.get("status_theory") else "",
            ])
        _print_table(n_rows, headers=["ok", "case", "dt", "dE_rel", "dh_rel", "status", "theory"])

        newton_suite_block: Dict[str, Any] = {
            "suite": "newton",
            "ok": bool(ok_newton),
            "n_cases": int(len(newton_cases)),
            "results": newton_results,
        }

        conv_reports: List[Dict[str, Any]] = []
        conv_ok = True
        if args.convergence:
            for c in newton_cases:
                cr = run_convergence_newton_one_case(
                    c,
                    plotdir if (args.plots or args.convergence) else None,
                    rigorous=bool(args.conv_rigorous),
                )
                conv_reports.append(cr)
                conv_ok = conv_ok and bool(cr.get("passed", False))

            _print_header(f"Newton convergence: ok={conv_ok} groups={len(conv_reports)}")

            c_rows: List[List[str]] = []
            for g in conv_reports:
                tag = "PASS" if g["passed"] else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
                dts = g.get("dt_effective", [])
                dts_str = ", ".join([f"{float(dt):.2e}" for dt in dts]) if dts else "-"
                c_rows.append([
                    tag,
                    str(g["name"]),
                    dts_str,
                    fmt_e(g.get("e_dt"), width=12),
                    fmt_e(g.get("e_dt2"), width=12),
                    fmt_f(g.get("p_obs"), width=8, prec=3),
                    fmt_e(g.get("abs_err_proxy"), width=12),
                    fmt_e(g.get("rel_err_proxy"), width=12),
                    short_msg(str(g.get("reason", ""))),
                ])
            _print_table(
                c_rows,
                headers=["ok", "case", "dt_eff (dt,dt/2,dt/4)", "e_dt", "e_dt2", "p_obs", "abs_err", "rel_err", "reason"],
            )

            newton_suite_block["ok_convergence"] = bool(conv_ok)
            newton_suite_block["convergence"] = conv_reports
            newton_suite_block["ok_total"] = bool(ok_newton and conv_ok)

        report["suites"].append(newton_suite_block)

    # ----------------------------
    # Schwarzschild (base suite)
    # ----------------------------
    if "schwarzschild" in suites:
        schw_cases = suites["schwarzschild"]["cases"]
        schw_results: List[Dict[str, Any]] = []
        ok_schw_cases = True

        for c in schw_cases:
            rr = validate_schw(
                c,
                plotdir if args.plots else None,
                time_plotdir if args.plots else None,
            )
            rr["_sig"] = schw_signature(c)
            schw_results.append(rr)
            ok_schw_cases = ok_schw_cases and bool(rr["passed"])

        conv = check_convergence_schw(schw_results, abs_tol=1e-9, rel_tol=0.25)
        conv_ok_s = all(bool(x["passed"]) for x in conv) if conv else False

        events_conv = check_convergence_events_schw(schw_results, abs_tol_factor=2.0, rel_tol=0.0)
        events_conv_ok = all(bool(x["passed"]) for x in events_conv) if events_conv else False

        ok_schw_total = bool(ok_schw_cases and conv_ok_s and events_conv_ok)

        _print_header(
            f"Schwarzschild suite: ok={ok_schw_total} "
            f"(cases_ok={ok_schw_cases}, conv_ok={conv_ok_s}, events_conv_ok={events_conv_ok}) cases={len(schw_cases)}"
        )

        s_rows: List[List[str]] = []
        for r in schw_results:
            s_rows.append([
                "PASS" if r["passed"] else "FAIL",
                str(r["name"]),
                f"{float(r['dt']):.1e}",
                fmt_f(r.get("r_min"), width=10, prec=6),
                fmt_f(r.get("r_end"), width=10, prec=6),
                fmt_e(r.get("constraint_abs_max"), width=12),
                fmt_e(r.get("norm_u_abs_max"), width=12),
                fmt_e(r.get("norm_u_abs_max_fd"), width=12),
                str(r.get("status", "")),
                str(r.get("events_compact", "") or ""),
                short_msg(str(r.get("message", ""))),
            ])
        _print_table(
            s_rows,
            headers=["ok", "case", "dt", "r_min", "r_end", "eps_max", "norm_u", "norm_u_fd", "status", "events", "msg"],
        )

        _print_header("Schwarzschild events (per run)")
        any_events = False
        for r in schw_results:
            evs = r.get("events", []) or []
            if evs:
                any_events = True
                print(f"{r['name']} (dt={float(r['dt']):.2e}): {r.get('events_compact','')}")
        if not any_events:
            print("No events detected in these runs.")

        _print_header("Schwarzschild time-dilation checks (t(τ), v(τ), dt/dτ, dv/dτ)")
        td_rows: List[List[str]] = []
        for r in schw_results:
            td_rows.append([
                "OK" if r["passed"] else "WARN/FAIL",
                str(r["name"]),
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
        _print_table(
            td_rows,
            headers=["ok", "case", "t", "t_finite", "t_mono", "dt_rel", "dt_abs",
                     "v", "v_finite", "v_mono", "dv_rel", "dv_abs", "mask_n"],
        )

        _print_header("Schwarzschild convergence: norm_u_abs_max should not increase when dt decreases")
        if not conv:
            print("No comparable groups found. Need >=2 cases with same physics and different dt.")
        else:
            c_rows2: List[List[str]] = []
            for g in conv:
                tag = "PASS" if g["passed"] else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
                dts = ", ".join([f"{float(dt):.2e}" for dt in g["dts"]])
                nus = ", ".join(["None" if v is None else f"{float(v):.3e}" for v in g["norm_u_abs_max"]])
                c_rows2.append([tag, dts, nus, ", ".join(g["cases"])])
            _print_table(c_rows2, headers=["ok", "dt (big->small)", "norm_u_abs_max", "cases"])

            for g in conv:
                if g.get("violations"):
                    for v in g["violations"]:
                        print(
                            f"violation: dt {float(v['dt_big']):.2e}->{float(v['dt_small']):.2e} "
                            f"norm_u {float(v['nu_big']):.3e}->{float(v['nu_small']):.3e} "
                            f"(abs_tol={float(v['abs_tol']):.1e}, rel_tol={float(v['rel_tol']):.2f})"
                        )

        report["suites"].append({
            "suite": "schwarzschild",
            "ok": bool(ok_schw_total),
            "ok_cases": bool(ok_schw_cases),
            "ok_convergence": bool(conv_ok_s),
            "ok_events_convergence": bool(events_conv_ok),
            "n_cases": int(len(schw_cases)),
            "results": schw_results,
            "convergence": conv,
            "events_convergence": events_conv,
        })

    # ----------------------------
    # Kerr Equatorial (base suite)
    # ----------------------------
    if "kerr_equatorial" in suites or "kerr" in suites:
        kerr_suite_name = "kerr_equatorial" if "kerr_equatorial" in suites else "kerr"
        kerr_cases = suites[kerr_suite_name]["cases"]
        kerr_results: List[Dict[str, Any]] = []
        ok_kerr_cases = True

        for c in kerr_cases:
            rr = validate_kerr(
                c,
                plotdir if args.plots else None,
                time_plotdir if args.plots else None,
            )
            rr["_sig"] = kerr_signature(c)
            kerr_results.append(rr)
            ok_kerr_cases = ok_kerr_cases and bool(rr["passed"])

        # Podemos reutilizar a mesma lógica de convergência de Schwarzschild!
        conv_k = check_convergence_schw(kerr_results, abs_tol=1e-9, rel_tol=0.25)
        conv_ok_k = all(bool(x["passed"]) for x in conv_k) if conv_k else False

        events_conv_k = check_convergence_events_schw(kerr_results, abs_tol_factor=2.0, rel_tol=0.0)
        events_conv_ok_k = all(bool(x["passed"]) for x in events_conv_k) if events_conv_k else False

        ok_kerr_total = bool(ok_kerr_cases and conv_ok_k and events_conv_ok_k)

        _print_header(
            f"Kerr Equatorial suite: ok={ok_kerr_total} "
            f"(cases_ok={ok_kerr_cases}, conv_ok={conv_ok_k}, events_conv_ok={events_conv_ok_k}) cases={len(kerr_cases)}"
        )

        k_rows: List[List[str]] = []
        for r in kerr_results:
            k_rows.append([
                "PASS" if r["passed"] else "FAIL",
                str(r["name"]),
                f"{float(r.get('a', 0.0)):.2f}",
                f"{float(r['dt']):.1e}",
                fmt_f(r.get("r_min"), width=10, prec=6),
                fmt_e(r.get("constraint_abs_max"), width=12),
                fmt_e(r.get("norm_u_abs_max"), width=12),
                str(r.get("status", "")),
                str(r.get("events_compact", "") or ""),
                short_msg(str(r.get("message", ""))),
            ])
        _print_table(
            k_rows,
            headers=["ok", "case", "a", "dt", "r_min", "eps_max", "norm_u", "status", "events", "msg"],
        )

        report["suites"].append({
            "suite": kerr_suite_name,
            "ok": bool(ok_kerr_total),
            "ok_cases": bool(ok_kerr_cases),
            "n_cases": int(len(kerr_cases)),
            "results": kerr_results,
            "convergence": conv_k,
            "events_convergence": events_conv_k,
        })

    # ----------------------------
    # Perihelion Precession suite
    # ----------------------------
    if "perihelion_precession" in suites:
        prec_cases = suites["perihelion_precession"]["cases"]
        prec_results: List[Dict[str, Any]] = []
        ok_prec = True

        for c in prec_cases:
            rr = validate_schw_perihelion(
                c,
                plotdir if args.plots else None,
                time_plotdir if args.plots else None,
            )
            rr["_sig"] = schw_signature(c)
            prec_results.append(rr)
            ok_prec = ok_prec and bool(rr["passed"])

        _print_header(
            f"Perihelion precession suite: ok={ok_prec} cases={len(prec_cases)}"
        )

        p_rows: List[List[str]] = []
        for r in prec_results:
            dphi_mean    = r.get("precession_delta_phi_mean")
            dphi_theory  = r.get("precession_delta_phi_exact")
            dphi_pn_disp = r.get("precession_delta_phi_pn")
            rel_err      = r.get("precession_rel_err")
            consist      = r.get("precession_consistency")
            n_orb        = r.get("precession_n_orbits", 0)
            used_exact   = r.get("precession_used_exact", False)
            th_tag       = "exact" if used_exact else "PN"
            p_rows.append([
                "PASS" if r["passed"] else "FAIL",
                str(r["name"]),
                f"{float(r['dt']):.1e}",
                str(int(n_orb)),
                fmt_e(dphi_mean, width=10),
                fmt_e(dphi_theory, width=10),
                fmt_e(dphi_pn_disp, width=10),
                th_tag,
                fmt_e(rel_err, width=10),
                fmt_e(consist, width=8),
                short_msg(str(r.get("message", ""))),
            ])
        _print_table(
            p_rows,
            headers=[
                "ok", "case", "dt", "n_orb",
                "Δφ_sim", "Δφ_exact", "Δφ_PN", "th",
                "rel_err", "consist", "msg",
            ],
        )

        # Sanity display: expected precession values
        _print_header("Perihelion precession — expected values (reference)")
        print(
            f"{'case':<42} {'Δφ_PN (rad)':>14} {'Δφ_PN (arcsec/orb)':>20}"
        )
        for r in prec_results:
            dphi_pn_val  = r.get("precession_delta_phi_pn", 0.0) or 0.0
            arcsec       = float(dphi_pn_val) * (180.0 * 3600.0 / 3.14159265358979)
            dphi_th_val  = r.get("precession_delta_phi_theory")
            th_str       = f"{dphi_th_val:.6f}" if dphi_th_val is not None else "N/A"
            print(
                f"  {r['name']:<40} Δφ_PN={float(dphi_pn_val):.6f}  "
                f"({arcsec:,.1f}\")  Δφ_exact={th_str}"
            )

        report["suites"].append({
            "suite": "perihelion_precession",
            "ok": bool(ok_prec),
            "n_cases": int(len(prec_cases)),
            "results": prec_results,
        })

    # ----------------------------
    # ISCO stability suite
    # ----------------------------
    if "isco" in suites:
        isco_cases = suites["isco"]["cases"]
        isco_results: List[Dict[str, Any]] = []
        ok_isco = True

        for c in isco_cases:
            rr = validate_schw_isco(
                c,
                plotdir if args.plots else None,
                time_plotdir if args.plots else None,
            )
            isco_results.append(rr)
            ok_isco = ok_isco and bool(rr["passed"])

        _print_header(
            f"ISCO stability suite: ok={ok_isco} cases={len(isco_cases)}"
        )

        i_rows: List[List[str]] = []
        for r in isco_results:
            i_rows.append([
                "PASS" if r["passed"] else "FAIL",
                str(r["name"]),
                f"{float(r['dt']):.1e}",
                fmt_f(r.get("isco_r_start"), width=8, prec=4),
                fmt_f(r.get("isco_r_M"), width=8, prec=4),
                "stable" if r.get("isco_theory_stable") else "unstable",
                str(r.get("status", "")),
                "ok" if r.get("isco_ok") else "FAIL",
                short_msg(str(r.get("message", ""))),
            ])
        _print_table(
            i_rows,
            headers=[
                "ok", "case", "dt",
                "r_start", "r_isco",
                "theory", "status",
                "isco_check", "msg",
            ],
        )

        print(f"\n  r_ISCO = 6M  (in geometric units G=c=1)")
        print(f"  r > 6M → stable orbit (BOUND expected)")
        print(f"  r < 6M → unstable orbit (CAPTURE expected)")

        report["suites"].append({
            "suite": "isco",
            "ok": bool(ok_isco),
            "n_cases": int(len(isco_cases)),
            "results": isco_results,
        })

    # ----------------------------
    # Salva relatório
    # ----------------------------
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