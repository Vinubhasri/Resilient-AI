"""
backend/routes.py — Multi-page routing with auth
"""
import os, io, time, json
from functools import wraps
from flask import (jsonify, request, render_template,
                   send_file, redirect, url_for, session, flash)
from backend.dnn_engine import DNN
from backend.repair_engine import RepairEngine
from backend.model_loader import load_model_from_source, validate_source, ModelParseError
from backend.report_generator import generate_html_report, generate_json_report

# ── Simple user store (demo — replace with DB in production) ──────────────────
USERS = {}  # {username: {password, created_at}}

_dnn = None
_engine = None

def _get_or_create_dnn():
    global _dnn, _engine
    if _dnn is None:
        _dnn = DNN()
        _engine = RepairEngine(_dnn)
        for _ in range(40):
            loss, acc = _dnn.train_step()
            _dnn.history.append({"loss": round(loss,4), "acc": round(acc*100,2)})
    return _dnn

def _set_dnn(dnn):
    global _dnn, _engine
    _dnn = dnn
    _engine = RepairEngine(_dnn)

def _allowed(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() == "py"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def register_routes(app):

    # ══════════════════════════════════════════════════════════════════════
    # AUTH PAGES
    # ══════════════════════════════════════════════════════════════════════

    @app.route("/")
    def home():
        if "user" in session:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET","POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username","").strip()
            password = request.form.get("password","").strip()
            if username in USERS and USERS[username]["password"] == password:
                session["user"] = username
                return redirect(url_for("dashboard"))
            return render_template("login.html", error="Invalid credentials.")
        return render_template("login.html", error=None)

    @app.route("/signup", methods=["GET","POST"])
    def signup():
        if request.method == "POST":
            username = request.form.get("username","").strip()
            password = request.form.get("password","").strip()
            confirm  = request.form.get("confirm","").strip()
            if not username or not password:
                return render_template("signup.html", error="All fields required.")
            if password != confirm:
                return render_template("signup.html", error="Passwords do not match.")
            if username in USERS:
                return render_template("signup.html", error="Username already exists.")
            USERS[username] = {"password": password, "created_at": time.strftime("%Y-%m-%d %H:%M")}
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("signup.html", error=None)

    @app.route("/logout")
    def logout():
        session.pop("user", None)
        return redirect(url_for("login"))

    # ══════════════════════════════════════════════════════════════════════
    # MAIN PAGES (login required)
    # ══════════════════════════════════════════════════════════════════════

    @app.route("/dashboard")
    @login_required
    def dashboard():
        dnn = _get_or_create_dnn()
        state = dnn.get_state()
        return render_template("dashboard.html", state=state, user=session["user"])

    @app.route("/upload", methods=["GET","POST"])
    @login_required
    def upload_model():
        if request.method == "POST":
            if "model_file" not in request.files:
                return render_template("upload.html", error="No file provided.", success=None, user=session["user"])
            f = request.files["model_file"]
            if not f.filename or not _allowed(f.filename):
                return render_template("upload.html", error="Only .py files allowed.", success=None, user=session["user"])
            source = f.read().decode("utf-8", errors="replace")
            validation = validate_source(source)
            if not validation["valid"]:
                return render_template("upload.html", error=validation["error"], success=None, user=session["user"])
            try:
                dnn = load_model_from_source(source, f.filename)
                _set_dnn(dnn)

                # ── Auto-repair on upload if issues detected ──────────────
                issues_before = dnn.health_check()
                state_before  = dnn.get_state()
                auto_repair_result = None

                if issues_before:
                    engine = RepairEngine(dnn)
                    repair_logs = engine.auto_repair()
                    # Fine-tune after repair
                    for _ in range(20):
                        loss, acc = dnn.train_step()
                        dnn.history.append({"loss": round(loss,4), "acc": round(acc*100,2)})
                    issues_after = dnn.health_check()
                    auto_repair_result = {
                        "triggered":      True,
                        "issues_before":  issues_before,
                        "issues_after":   issues_after,
                        "logs":           repair_logs,
                        "loss_before":    state_before["loss"],
                        "acc_before":     state_before["accuracy"],
                        "loss_after":     round(dnn.get_state()["loss"], 4),
                        "acc_after":      round(dnn.get_state()["accuracy"], 2),
                        "fixed_count":    len(issues_before) - len(issues_after),
                    }

                return render_template("upload.html", error=None,
                    success={"name": dnn.name, "arch": dnn.arch, "file": f.filename},
                    auto_repair=auto_repair_result,
                    user=session["user"])
            except Exception as e:
                return render_template("upload.html", error=str(e), success=None, user=session["user"])
        return render_template("upload.html", error=None, success=None, auto_repair=None, user=session["user"])

    @app.route("/inject", methods=["GET","POST"])
    @login_required
    def inject_error():
        dnn = _get_or_create_dnn()
        result = None
        if request.method == "POST":
            layer_idx  = int(request.form.get("layer", 0))
            error_type = request.form.get("error_type", "weight_explosion")
            if 0 <= layer_idx < len(dnn.layers):
                dnn.layers[layer_idx].inject_error(error_type)
                dnn.repaired = False
                result = {"layer": dnn.layers[layer_idx].name, "error": error_type}
        return render_template("inject.html", dnn=dnn, result=result, user=session["user"])

    @app.route("/repair", methods=["GET","POST"])
    @login_required
    def repair():
        global _engine
        dnn = _get_or_create_dnn()
        result = None
        if request.method == "POST":
            strategy = request.form.get("strategy", "auto")
            layer_i  = request.form.get("layer", None)
            if layer_i is not None and layer_i != "all":
                layer_i = int(layer_i)
            else:
                layer_i = None
            logs = _engine.apply(strategy, layer_i)
            for _ in range(20):
                loss, acc = dnn.train_step()
                dnn.history.append({"loss": round(loss,4), "acc": round(acc*100,2)})
            result = {"logs": logs, "strategy": strategy}
        return render_template("repair.html", dnn=dnn, result=result, user=session["user"])

    @app.route("/graphs")
    @login_required
    def graphs():
        dnn = _get_or_create_dnn()
        state = dnn.get_state()
        return render_template("graphs.html", state=state, user=session["user"])

    @app.route("/reports")
    @login_required
    def reports():
        dnn = _get_or_create_dnn()
        state = dnn.get_state()
        return render_template("reports.html", state=state, user=session["user"])

    # ══════════════════════════════════════════════════════════════════════
    # API ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    @app.route("/api/state")
    @login_required
    def api_state():
        return jsonify(_get_or_create_dnn().get_state())

    @app.route("/api/train", methods=["POST"])
    @login_required
    def api_train():
        dnn   = _get_or_create_dnn()
        steps = min(int(request.json.get("steps", 25)), 200)
        for _ in range(steps):
            loss, acc = dnn.train_step()
            dnn.history.append({"loss": round(loss,4), "acc": round(acc*100,2)})
        return jsonify(dnn.get_state())

    @app.route("/api/reset", methods=["POST"])
    @login_required
    def api_reset():
        dnn = DNN()
        _set_dnn(dnn)
        for _ in range(40):
            loss, acc = dnn.train_step()
            dnn.history.append({"loss": round(loss,4), "acc": round(acc*100,2)})
        return jsonify(dnn.get_state())

    @app.route("/api/download_report/html")
    @login_required
    def download_html():
        dnn  = _get_or_create_dnn()
        html = generate_html_report(dnn.get_state())
        buf  = io.BytesIO(html.encode("utf-8"))
        buf.seek(0)
        return send_file(buf, mimetype="text/html", as_attachment=True,
                         download_name=f"dnn_report_{time.strftime('%Y%m%d_%H%M%S')}.html")

    @app.route("/api/download_report/json")
    @login_required
    def download_json():
        dnn  = _get_or_create_dnn()
        data = generate_json_report(dnn.get_state())
        buf  = io.BytesIO(data.encode("utf-8"))
        buf.seek(0)
        return send_file(buf, mimetype="application/json", as_attachment=True,
                         download_name=f"dnn_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
