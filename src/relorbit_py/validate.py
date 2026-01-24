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
    run_convergence_newton_one_case,
    schw_signature,
    check_convergence_schw,
    check_convergence_events_schw,
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
        help="Use stricter defaults for convergence criteria (can be overridden per-case in YAML criteria.*).",
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

    # ----------------------------
    # Newton
    # ----------------------------
    newton_cases = cfg["suites"]["newton"]["cases"]
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
    # Schwarzschild
    # ----------------------------
    schw_cases = cfg["suites"]["schwarzschild"]["cases"]
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
            fmt_e(r.get("norm_u_abs_max"), width=12),        # primary (theory preferred)
            fmt_e(r.get("norm_u_abs_max_fd"), width=12),     # diagnostic
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

    _print_header("Schwarzschild convergence: norm_u_abs_max should not increase when dt decreases (with tolerance)")
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

    _print_header("Schwarzschild convergence: event times should change little when dt decreases")
    if not events_conv:
        print("No comparable groups found. Need >=2 cases with same physics and different dt.")
    else:
        e_rows: List[List[str]] = []
        for g in events_conv:
            tag = "PASS" if g["passed"] else (
                "SKIP" if g.get("skipped") else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            )
            dts = ", ".join([f"{float(dt):.2e}" for dt in g["dts"]])
            reason = str(g.get("reason", "")) if (g.get("skipped") or g.get("inconclusive")) else ""
            e_rows.append([tag, dts, ", ".join(g["cases"]), reason])
        _print_table(e_rows, headers=["ok", "dt (big->small)", "cases", "reason"])

        for g in events_conv:
            for mm in g.get("mismatches", []) or []:
                print(
                    "mismatch: "
                    f"{mm.get('kind','?')} count dt {float(mm.get('dt_big',0.0)):.2e}->{float(mm.get('dt_small',0.0)):.2e} "
                    f"{mm.get('count_big','?')}->{mm.get('count_small','?')}"
                )
            for v in g.get("violations", []) or []:
                print(
                    f"violation: {v['kind']}[{v['occurrence']}] dt {float(v['dt_big']):.2e}->{float(v['dt_small']):.2e} "
                    f"tau {float(v['tau_big']):.6g}->{float(v['tau_small']):.6g} "
                    f"abs_err={float(v['abs_err']):.3e} allowed={float(v['allowed']):.3e} "
                    f"(abs_tol={float(v['abs_tol']):.3e}, rel_tol={float(v['rel_tol']):.2f})"
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
