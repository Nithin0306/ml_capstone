import os
import json
import joblib
import numpy as np
import pandas as pd
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

print(f"Loading models from: {MODELS_DIR}...")
class_artifact = joblib.load(os.path.join(MODELS_DIR, "classification_rf.joblib"))
clust_artifact = joblib.load(os.path.join(MODELS_DIR, "clustering_kmeans.joblib"))
reg_artifact   = joblib.load(os.path.join(MODELS_DIR, "regression_rf.joblib"))

with open(os.path.join(MODELS_DIR, "leaderboard.json")) as f:
    leaderboard_data = json.load(f)

print("All models and metadata loaded successfully into memory.")

class MLServerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({"status": "healthy", "service": "ML Capstone Inference Engine"})
        elif path == "/api/leaderboard":
            self._send_json(leaderboard_data)
        elif path == "/api/models/meta":
            self._send_json({
                "classification": class_artifact["metrics"],
                "clustering": clust_artifact["metrics"],
                "regression": reg_artifact["metrics"]
            })
        elif path == "/api/clustering/background":
            self._send_json({
                "pca_points": clust_artifact["pca_background"],
                "labels": clust_artifact["labels_background"],
                "cluster_stats": clust_artifact["metrics"]["cluster_stats"]
            })
        else:
            # Fallback to serving static files from frontend if accessed directly
            frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
            if os.path.exists(frontend_dir):
                self.directory = frontend_dir
                return super().do_GET()
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len)
        try:
            data = json.loads(post_body.decode('utf-8')) if post_body else {}
        except Exception as e:
            return self._send_json({"error": "Invalid JSON payload", "details": str(e)}, status=400)

        if path == "/api/predict/classification":
            self._handle_classification(data)
        elif path == "/api/predict/clustering":
            self._handle_clustering(data)
        elif path == "/api/predict/regression":
            self._handle_regression(data)
        else:
            self.send_error(404, "API endpoint not found")

    def _send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def _handle_classification(self, data):
        # Default fallback values for credit approval
        # num_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
        # cat_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
        num_vals = [
            float(data.get('A2', 31.5)),     # Age
            float(data.get('A3', 4.5)),      # Debt
            float(data.get('A8', 2.0)),      # Years Employed
            float(data.get('A11', 1.0)),     # Credit Score / Active Lines
            float(data.get('A14', 180.0)),   # Zip Code / Numerical
            float(data.get('A15', 500.0))    # Income
        ]
        cat_vals = [
            str(data.get('A1', 'b')),
            str(data.get('A4', 'u')),
            str(data.get('A5', 'g')),
            str(data.get('A6', 'c')),
            str(data.get('A7', 'v')),
            str(data.get('A9', 't')),        # Prior Default: t = prior default, f = no prior default
            str(data.get('A10', 't')),       # Employed: t = yes, f = no
            str(data.get('A12', 'f')),
            str(data.get('A13', 'g'))
        ]

        num_imp = class_artifact["num_imputer"].transform([num_vals])
        num_scaled = class_artifact["scaler"].transform(num_imp)
        cat_imp = class_artifact["cat_imputer"].transform([cat_vals])
        cat_ohe = class_artifact["ohe"].transform(cat_imp)

        X_in = np.hstack([num_scaled, cat_ohe])
        prob = class_artifact["model"].predict_proba(X_in)[0][1]
        pred = int(prob >= 0.5)

        # Top driving feature impacts
        top_importances = sorted(
            zip(class_artifact["feature_names"], class_artifact["importances"]),
            key=lambda x: x[1], reverse=True
        )[:5]

        self._send_json({
            "prediction": "APPROVED" if pred == 1 else "REJECTED",
            "approved": pred == 1,
            "approval_probability": round(float(prob) * 100, 2),
            "risk_score": round((1.0 - float(prob)) * 100, 2),
            "top_drivers": [{"feature": f, "weight": round(w * 100, 2)} for f, w in top_importances]
        })

    def _handle_clustering(self, data):
        # sensor_cols = 19 sensor measurements
        sensor_cols = clust_artifact["sensor_cols"]
        vals = []
        for col in sensor_cols:
            val = data.get(col, None)
            vals.append(float(val) if val is not None else np.nan)

        vals_arr = np.array([vals])
        vals_imp = clust_artifact["imputer"].transform(vals_arr)
        vals_scaled = clust_artifact["scaler"].transform(vals_imp)

        cluster_id = int(clust_artifact["model"].predict(vals_scaled)[0])
        pca_coords = clust_artifact["pca"].transform(vals_scaled)[0].tolist()

        c_info = clust_artifact["metrics"]["cluster_stats"].get(str(cluster_id), {})

        self._send_json({
            "assigned_cluster": cluster_id,
            "cluster_name": c_info.get("name", f"Cluster {cluster_id}"),
            "hazard_rate_percent": c_info.get("rate", 0.0),
            "risk_level": "ELEVATED RISK" if cluster_id == 0 else "NORMAL OPERATION",
            "pca_coordinates": [round(pca_coords[0], 3), round(pca_coords[1], 3)]
        })

    def _handle_regression(self, data):
        features = reg_artifact["features"]
        vals = []
        for col in features:
            vals.append(float(data.get(col, 100.0)))

        vals_arr = np.array([vals])
        vals_imp = reg_artifact["imputer"].transform(vals_arr)
        vals_scaled = reg_artifact["scaler"].transform(vals_imp)

        pred_val = float(reg_artifact["model"].predict(vals_scaled)[0])

        self._send_json({
            "target": reg_artifact["metrics"]["target"],
            "predicted_value": round(pred_val, 2),
            "unit": "μg/m³",
            "air_quality_category": "Good (< 5)" if pred_val < 5 else ("Moderate (5-10)" if pred_val < 10 else "High Pollution (> 10)")
        })

if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, MLServerHandler)
    print(f"ML Capstone Server listening on http://localhost:{PORT}")
    httpd.serve_forever()
