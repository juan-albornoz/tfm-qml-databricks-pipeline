"""Auditoria de sensibilidad del Live Predictor: replica exactamente la ruta de
inferencia de app.py (predict_real) y barre cada variable."""
import json, numpy as np, onnxruntime as ort
from pathlib import Path

# Relativa al propio script: audit_scripts/ -> raiz del repo -> streamlit/models.
M = Path(__file__).resolve().parent.parent / "streamlit" / "models"
sc = json.loads((M / "scaler_correcto.json").read_text())
med = json.loads((M / "medianas_correctas.json").read_text())
FEATS = sc["features"]; MEAN = np.array(sc["mean"]); SCALE = np.array(sc["scale"])
s_lgbm = ort.InferenceSession(str(M / "lgbm_final.onnx"))
s_svm = ort.InferenceSession(str(M / "svm_final.onnx"))

QSVM = {
    "LBXGH":   (4.0, 15.0, 5.7), "LBXGLU":   (50, 300, 100),
    "RIDAGEYR":(18, 80, 45),     "LBDLDL":   (40, 250, 110),
    "BMXWAIST":(60, 150, 95),    "LBXIN":    (2, 60, 10),
    "BMXLEG":  (30, 50, 40),     "BMXBMI":   (15, 60, 27),
}
DEFAULTS = {k: v[2] for k, v in QSVM.items()}

def predict(ov):
    x = np.array([med[f] for f in FEATS], dtype=np.float64)
    for k, v in ov.items():
        if k in FEATS:
            x[FEATS.index(k)] = v
    p_l = float(s_lgbm.run(None, {"float_input": x.reshape(1,-1).astype(np.float32)})[1][0][1])
    xs = ((x - MEAN) / SCALE).reshape(1,-1).astype(np.float32)
    p_s = float(s_svm.run(None, {"float_input": xs})[1][0][1])
    return p_l, p_s

print("=== BASELINE (defaults del slider) ===")
pl, ps = predict(DEFAULTS)
print(f"LGBM={pl:.4f}  SVM={ps:.4f}\n")

print("=== HbA1c: barrido fino 4.0 -> 9.0 (resto en defaults) ===")
prev = None
for v in np.arange(4.0, 9.01, 0.1):
    ov = dict(DEFAULTS); ov["LBXGH"] = round(float(v), 1)
    pl, ps = predict(ov)
    d = "" if prev is None else f"  dLGBM={pl-prev:+.4f}"
    print(f"HbA1c={v:4.1f}  LGBM={pl:.4f}  SVM={ps:.4f}{d}")
    prev = pl

print("\n=== HbA1c: micro-barrido 5.75 -> 6.35 paso 0.01 (localizar el escalon) ===")
prev = None
for v in np.arange(5.75, 6.351, 0.01):
    ov = dict(DEFAULTS); ov["LBXGH"] = float(v)
    pl, ps = predict(ov)
    d = "" if prev is None else f"  salto={pl-prev:+.4f}"
    print(f"HbA1c={v:5.2f}  LGBM={pl:.4f}  SVM={ps:.4f}{d}")
    prev = pl

print("\n=== Barrido del resto de variables (min, p25, default, p75, max) ===")
for k, (lo, hi, dflt) in QSVM.items():
    if k == "LBXGH": continue
    print(f"\n-- {k} --")
    for v in np.linspace(lo, hi, 9):
        ov = dict(DEFAULTS); ov[k] = float(v)
        pl, ps = predict(ov)
        print(f"  {k}={v:8.2f}  LGBM={pl:.4f}  SVM={ps:.4f}")

print("\n=== Rango total de cada variable (efecto marginal aislado) ===")
for k, (lo, hi, dflt) in QSVM.items():
    vals = np.linspace(lo, hi, 60)
    pls, pss = [], []
    for v in vals:
        ov = dict(DEFAULTS); ov[k] = float(v)
        a, b = predict(ov); pls.append(a); pss.append(b)
    print(f"{k:9s} LGBM[{min(pls):.3f},{max(pls):.3f}] rango={max(pls)-min(pls):.3f} | "
          f"SVM[{min(pss):.3f},{max(pss):.3f}] rango={max(pss)-min(pss):.3f}")
