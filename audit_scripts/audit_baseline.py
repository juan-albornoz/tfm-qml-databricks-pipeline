"""Auditoria del VECTOR BASE: las 81 features no editables fijadas a la mediana."""
import json, numpy as np, onnxruntime as ort
from pathlib import Path
# Relativa al propio script: audit_scripts/ -> raiz del repo -> streamlit/models.
M = Path(__file__).resolve().parent.parent / "streamlit" / "models"
sc = json.loads((M/"scaler_correcto.json").read_text()); med = json.loads((M/"medianas_correctas.json").read_text())
F = sc["features"]; MEAN = np.array(sc["mean"]); SCALE = np.array(sc["scale"])
sl = ort.InferenceSession(str(M/"lgbm_final.onnx")); ss = ort.InferenceSession(str(M/"svm_final.onnx"))
DEF = {"LBXGH":5.7,"LBXGLU":100,"RIDAGEYR":45,"LBDLDL":110,"BMXWAIST":95,"LBXIN":10,"BMXLEG":40,"BMXBMI":27}

def pred(ov, base=None):
    x = np.array([med[f] for f in F], float) if base is None else base.copy()
    for k,v in ov.items():
        if k in F: x[F.index(k)] = v
    pl = float(sl.run(None,{"float_input":x.reshape(1,-1).astype(np.float32)})[1][0][1])
    ps = float(ss.run(None,{"float_input":(((x-MEAN)/SCALE).reshape(1,-1)).astype(np.float32)})[1][0][1])
    return pl, ps

print(f"n_features = {len(F)}  (app declara 89)\n")

print("=== A. Dummies one-hot en el vector base (mediana por columna) ===")
groups = {}
for f in F:
    if "_" in f and f.split("_")[-1].replace(".","").isdigit():
        groups.setdefault(f.rsplit("_",1)[0], []).append(f)
for g, cols in groups.items():
    s = sum(med[c] for c in cols)
    prev = {c: MEAN[F.index(c)] for c in cols}
    moda = max(prev, key=prev.get)
    flag = "  <<< SUMA 0: categoria INEXISTENTE" if s == 0 else ""
    print(f"  {g:12s} suma={s:.0f}  moda real={moda} (prev={prev[moda]:.3f}){flag}")

print("\n=== B. Impacto de arreglar el one-hot (usar la MODA en vez de la mediana) ===")
base_fix = np.array([med[f] for f in F], float)
for g, cols in groups.items():
    if sum(med[c] for c in cols) == 0:
        moda = max(cols, key=lambda c: MEAN[F.index(c)])
        base_fix[F.index(moda)] = 1.0
pl0, ps0 = pred(DEF)
pl1, ps1 = pred(DEF, base_fix)
print(f"  base actual (medianas):  LGBM={pl0:.4f}  SVM={ps0:.4f}")
print(f"  base corregida (modas):  LGBM={pl1:.4f}  SVM={ps1:.4f}")
print(f"  DESVIACION:              LGBM={pl1-pl0:+.4f}  SVM={ps1-ps0:+.4f}")

print("\n=== C. HbA1c 5.8 vs 6.3 con las dos bases ===")
for lbl, b in [("medianas", None), ("modas", base_fix)]:
    r = {}
    for h in (5.8, 6.3):
        o = dict(DEF); o["LBXGH"] = h
        r[h] = pred(o, b)
    print(f"  base {lbl:9s}: 5.8 -> LGBM {r[5.8][0]:.4f} | 6.3 -> LGBM {r[6.3][0]:.4f}"
          f"   (salto {r[6.3][0]-r[5.8][0]:+.4f})")

print("\n=== D. Sensibilidad al PESO MUESTRAL WTINT2YR (artefacto de diseno, no clinico) ===")
for w in [5000, 15000, 25697, 40000, 70000, 120000, 200000]:
    o = dict(DEF); o["WTINT2YR"] = w
    pl, ps = pred(o)
    print(f"  WTINT2YR={w:7d}  LGBM={pl:.4f}  SVM={ps:.4f}")

print("\n=== E. Coherencia LGBM vs SVM sobre perfiles clinicos tipicos ===")
perfiles = {
    "Sano joven":        dict(LBXGH=5.0, LBXGLU=85,  RIDAGEYR=30, LBDLDL=100, BMXWAIST=80,  LBXIN=6,  BMXLEG=42, BMXBMI=22),
    "Prediabetico":      dict(LBXGH=6.0, LBXGLU=115, RIDAGEYR=55, LBDLDL=130, BMXWAIST=105, LBXIN=18, BMXLEG=38, BMXBMI=31),
    "Diabetico franco":  dict(LBXGH=8.5, LBXGLU=180, RIDAGEYR=62, LBDLDL=95,  BMXWAIST=120, LBXIN=25, BMXLEG=37, BMXBMI=36),
    "Diab. descontrol.": dict(LBXGH=11.0,LBXGLU=250, RIDAGEYR=68, LBDLDL=90,  BMXWAIST=130, LBXIN=35, BMXLEG=36, BMXBMI=40),
}
for n, p in perfiles.items():
    pl, ps = pred(p)
    cat = "Bajo" if pl < .33 else ("Moderado" if pl < .5 else "Elevado")
    print(f"  {n:18s} LGBM={pl:.4f} [{cat:9s}]  SVM={ps:.4f}  |diferencia|={abs(pl-ps):.4f}")

print("\n=== F. Rango de los sliders vs distribucion de entrenamiento (media +/- 3 sd) ===")
RANGES = {"LBXGH":(4.0,15.0),"LBXGLU":(50,300),"RIDAGEYR":(18,80),"LBDLDL":(40,250),
          "BMXWAIST":(60,150),"LBXIN":(2,60),"BMXLEG":(30,50),"BMXBMI":(15,60)}
for k,(lo,hi) in RANGES.items():
    i = F.index(k); mu, sd = MEAN[i], SCALE[i]
    print(f"  {k:9s} slider[{lo:6.1f},{hi:6.1f}]  train media={mu:7.2f} sd={sd:6.2f}"
          f"  +/-3sd=[{mu-3*sd:7.2f},{mu+3*sd:7.2f}]  mediana={med[k]}"
          f"  z(min)={(lo-mu)/sd:+.2f} z(max)={(hi-mu)/sd:+.2f}")
