"""
DNN Automated Repair System — Multi-Page Edition
Run: python app.py  |  Open: http://localhost:5000
"""
from flask import Flask
from backend.routes import register_routes
import os

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
app.secret_key = "dnn-repair-secret-2024"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["REPORTS_FOLDER"] = "reports"

register_routes(app)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    print("\n" + "═" * 56)
    print("  DNN REPAIR SYSTEM — Multi-Page Edition")
    print("═" * 56)
    print("  → http://localhost:5000")
    print("═" * 56 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
