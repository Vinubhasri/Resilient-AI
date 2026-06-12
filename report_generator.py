"""
backend/report_generator.py
────────────────────────────
Generates detailed HTML and plain-text error/repair reports
that the user can download from the dashboard.
"""

import time
import json


def _health_color(healthy: bool) -> str:
    return "#00FF88" if healthy else "#FF4560"


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
        f'padding:2px 10px;border-radius:100px;font-size:11px;font-family:monospace">'
        f'{text}</span>'
    )


STRATEGY_COLORS = {
    "Weight Clipping"  : "#00E5FF",
    "Xavier Reinit"    : "#B06AFF",
    "Gradient Norm"    : "#FFB800",
    "Noise Denoising"  : "#00FF88",
    "Bias Correction"  : "#FF8C42",
    "Model Loader"     : "#7A8AAD",
    "Auto Repair"      : "#00FF88",
}


def generate_html_report(state: dict) -> str:
    """Builds a self-contained HTML report string."""

    ts       = time.strftime("%Y-%m-%d %H:%M:%S")
    layers   = state.get("layers", [])
    issues   = state.get("issues", [])
    log      = state.get("repair_log", [])
    history  = state.get("history", [])
    name     = state.get("name", "DNN")
    arch     = state.get("arch", [])
    loss     = state.get("loss", "—")
    accuracy = state.get("accuracy", "—")
    repaired = state.get("repaired", False)
    src_file = state.get("source_file", "default model")

    # Accuracy trend
    if history:
        first_acc = history[0].get("acc", 0)
        last_acc  = history[-1].get("acc", 0)
        delta_acc = round(last_acc - first_acc, 2)
        trend     = f"+{delta_acc}%" if delta_acc >= 0 else f"{delta_acc}%"
    else:
        trend = "N/A"

    # Layer rows
    layer_rows = ""
    for i, l in enumerate(layers):
        status_badge = _badge("HEALTHY", "#00FF88") if l["healthy"] else _badge(l.get("error","ERROR"), "#FF4560")
        layer_rows += f"""
        <tr style="border-bottom:1px solid #232B40">
          <td style="padding:10px 14px;font-family:monospace;color:#00E5FF">{i}</td>
          <td style="padding:10px 14px">{l['name']}</td>
          <td style="padding:10px 14px;font-family:monospace">{l['shape'][0]}×{l['shape'][1]}</td>
          <td style="padding:10px 14px;font-family:monospace">{l['mean']:.5f}</td>
          <td style="padding:10px 14px;font-family:monospace">{l['std']:.5f}</td>
          <td style="padding:10px 14px;font-family:monospace">{l['max']:.5f}</td>
          <td style="padding:10px 14px;font-family:monospace">{l['zeros']*100:.1f}%</td>
          <td style="padding:10px 14px">{status_badge}</td>
        </tr>"""

    # Issues rows
    if issues:
        issue_rows = "".join(f"""
        <tr style="border-bottom:1px solid #2a1520">
          <td style="padding:8px 14px;color:#FF4560;font-family:monospace">⚠</td>
          <td style="padding:8px 14px;font-family:monospace;color:#FFB800">{iss[0]}</td>
          <td style="padding:8px 14px;color:#E8EDF8">{iss[1].replace('_',' ').title()}</td>
        </tr>""" for iss in issues)
    else:
        issue_rows = '<tr><td colspan="3" style="padding:12px 14px;color:#00FF88">✓ No issues detected</td></tr>'

    # Repair log rows
    if log:
        log_rows = "".join(f"""
        <tr style="border-bottom:1px solid #232B40">
          <td style="padding:8px 14px;font-family:monospace;color:#7A8AAD">{e['time']}</td>
          <td style="padding:8px 14px">{_badge(e['strategy'], STRATEGY_COLORS.get(e['strategy'],'#7A8AAD'))}</td>
          <td style="padding:8px 14px;font-family:monospace;color:#E8EDF8;font-size:12px">{e.get('layer','—')}</td>
          <td style="padding:8px 14px;font-size:12px;color:#7A8AAD">{e['message']}</td>
          <td style="padding:8px 14px;font-family:monospace;color:#FFB800">{e['before'] if e['before'] is not None else '—'}</td>
          <td style="padding:8px 14px;font-family:monospace;color:#00FF88">{e['after'] if e['after'] is not None else '—'}</td>
        </tr>""" for e in log)
    else:
        log_rows = '<tr><td colspan="6" style="padding:12px 14px;color:#7A8AAD;font-family:monospace">No repair actions recorded.</td></tr>'

    overall_status = _badge("REPAIRED", "#00FF88") if repaired else (_badge("ISSUES FOUND", "#FF4560") if issues else _badge("HEALTHY", "#00FF88"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DNN Repair Report — {name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0A0D14; color: #E8EDF8; font-family: 'DM Sans', sans-serif, Arial; padding: 40px; line-height: 1.6; }}
  h1 {{ font-size: 22px; font-weight: 600; color: #00E5FF; letter-spacing: 1px; }}
  h2 {{ font-size: 14px; font-family: monospace; color: #7A8AAD; letter-spacing: 2px; text-transform: uppercase; margin: 32px 0 12px; border-left: 3px solid #00E5FF; padding-left: 12px; }}
  table {{ width: 100%; border-collapse: collapse; background: #111520; border-radius: 8px; overflow: hidden; }}
  th {{ text-align: left; padding: 10px 14px; font-size: 11px; font-family: monospace; color: #7A8AAD; letter-spacing: 1px; background: #161C2D; border-bottom: 1px solid #232B40; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
  .stat {{ background: #111520; border: 1px solid #232B40; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat-val {{ font-family: monospace; font-size: 24px; font-weight: 700; }}
  .stat-lbl {{ font-size: 11px; color: #7A8AAD; margin-top: 4px; font-family: monospace; letter-spacing: 1px; }}
  .header-bar {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #232B40; }}
  .meta {{ font-size: 12px; color: #7A8AAD; font-family: monospace; }}
  footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #232B40; font-size: 11px; color: #7A8AAD; font-family: monospace; text-align: center; }}
</style>
</head>
<body>

<div class="header-bar">
  <div>
    <h1>DNN REPAIR REPORT</h1>
    <div class="meta">Model: {name} &nbsp;|&nbsp; Source: {src_file} &nbsp;|&nbsp; Generated: {ts}</div>
  </div>
  <div>{overall_status}</div>
</div>

<h2>Executive Summary</h2>
<div class="stat-grid">
  <div class="stat"><div class="stat-val" style="color:#00E5FF">{loss}</div><div class="stat-lbl">FINAL LOSS</div></div>
  <div class="stat"><div class="stat-val" style="color:#00FF88">{accuracy}%</div><div class="stat-lbl">ACCURACY</div></div>
  <div class="stat"><div class="stat-val" style="color:#FFB800">{len(issues)}</div><div class="stat-lbl">ISSUES FOUND</div></div>
  <div class="stat"><div class="stat-val" style="color:#B06AFF">{len(log)}</div><div class="stat-lbl">REPAIRS APPLIED</div></div>
</div>
<div style="background:#111520;border:1px solid #232B40;border-radius:8px;padding:14px;margin:12px 0;font-size:13px">
  <b style="color:#00E5FF">Architecture:</b> <span style="font-family:monospace">{' → '.join(str(n) for n in arch)}</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <b style="color:#00E5FF">Training Steps:</b> <span style="font-family:monospace">{len(history)}</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <b style="color:#00E5FF">Accuracy Trend:</b> <span style="font-family:monospace;color:#00FF88">{trend}</span>
</div>

<h2>Layer Analysis</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Layer</th><th>Shape</th>
      <th>Mean |W|</th><th>Std W</th><th>Max |W|</th><th>Zero %</th><th>Status</th>
    </tr>
  </thead>
  <tbody>{layer_rows}</tbody>
</table>

<h2>Detected Issues</h2>
<table>
  <thead><tr><th>Severity</th><th>Layer</th><th>Issue Type</th></tr></thead>
  <tbody>{issue_rows}</tbody>
</table>

<h2>Repair Log ({len(log)} actions)</h2>
<table>
  <thead>
    <tr><th>Time</th><th>Strategy</th><th>Layer</th><th>Description</th><th>Before</th><th>After</th></tr>
  </thead>
  <tbody>{log_rows}</tbody>
</table>

<footer>
  DNN Automated Repair System — Resilient AI &nbsp;|&nbsp;
  Report generated {ts} &nbsp;|&nbsp;
  {len(layers)} layers analysed &nbsp;|&nbsp;
  {len(log)} repair actions applied
</footer>
</body>
</html>"""


def generate_json_report(state: dict) -> str:
    """Returns a formatted JSON string for download."""
    report = {
        "report_meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name"  : state.get("name"),
            "source_file" : state.get("source_file"),
            "architecture": state.get("arch"),
        },
        "performance": {
            "final_loss"    : state.get("loss"),
            "final_accuracy": state.get("accuracy"),
            "training_steps": len(state.get("history", [])),
        },
        "issues_detected": state.get("issues", []),
        "layer_stats"    : state.get("layers", []),
        "repair_log"     : state.get("repair_log", []),
        "repaired"       : state.get("repaired", False),
    }
    return json.dumps(report, indent=2)
