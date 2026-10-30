"""Tests for the project service (run: pytest code/week-21/project). Uses FastAPI's in-process TestClient, no network."""
import os, json, numpy as np, pytest
from fastapi.testclient import TestClient
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE); import service
C = TestClient(service.app); ART = os.path.join(HERE, "artifacts"); X = np.load(os.path.join(ART, "golden_X.npy")); G = json.load(open(os.path.join(ART, "golden.json")))
def test_health(): r = C.get("/health"); assert r.status_code == 200 and r.json()["model_version"] == G["weights_sha"]
def test_golden_predictions():  # outputs must match the probabilities saved at export time (catches preprocessing / export drift)
    p = C.post("/predict", json={"rows": X.tolist()}).json()["p_benign"]; assert np.abs(np.array(p) - np.array(G["p"])).max() < 1e-5
def test_accuracy_floor(): assert G["test_acc"] >= 0.93   # release gate
@pytest.mark.parametrize("body", [{"rows": [[1.0, 2.0]]}, {"rows": "x"}, {"rows": []}, {}])
def test_rejects_bad_input(body): assert C.post("/predict", json=body).status_code == 422
def test_batch_limit(): assert C.post("/predict", json={"rows": [[0.0] * 30] * 513}).status_code == 422
def test_drift_alert_fires_and_stays_quiet():
    service.WIN.clear(); C.post("/predict", json={"rows": X.tolist()}); quiet = C.get("/metrics").json()["drift"]
    Xd = X.copy(); Xd[:, service.META["features"].index("mean area")] *= 1.3
    service.WIN.clear(); C.post("/predict", json={"rows": Xd.tolist()}); loud = C.get("/metrics").json()["drift"]
    assert not quiet["alert"] and loud["alert"] and loud["feature"].endswith("area")
def test_rejects_nan_sent_as_raw_json():  # strict JSON has no NaN, but Python's json parser accepts the literal NaN, so the server must check
    r = C.post("/predict", content="{\"rows\": [[" + ",".join(["NaN"] * 30) + "]]}", headers={"content-type": "application/json"}); assert r.status_code == 422
