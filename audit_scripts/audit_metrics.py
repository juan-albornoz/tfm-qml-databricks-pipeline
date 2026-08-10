"""Verificacion de las metricas HARDCODEADAS en app.py contra los .npy reales."""
import numpy as np
from pathlib import Path
# Relativa al propio script: audit_scripts/ -> raiz del repo -> streamlit/models.
M = Path(__file__).resolve().parent.parent / "streamlit" / "models"

DECL = {
    "lgbm": dict(auc=0.9485, f1_macro=0.6523, accuracy=0.7243, mcc=0.4566, tn=924, fp=423, fn=9, tp=211),
    "svm":  dict(auc=0.9377, f1_macro=0.8243, accuracy=0.9075, mcc=0.6539, tn=1250, fp=97, fn=48, tp=172),
    "qsvm": dict(auc=0.5493, f1_macro=0.4669, accuracy=0.8602, mcc=0.0625, tn=1347, fp=0, fn=219, tp=1),
}

def auc_of(y, s):
    o = np.argsort(-s); ys = y[o]
    tps = np.cumsum(ys); fps = np.cumsum(1-ys)
    tpr = tps/tps[-1]; fpr = fps/fps[-1]
    return np.trapezoid(np.concatenate([[0],tpr]), np.concatenate([[0],fpr]))

def rankdata(a):  # rangos medios ante empates, sin scipy
    o = np.argsort(a, kind="mergesort"); sa = a[o]
    r = np.empty(len(a), dtype=float); i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j+1] == sa[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r

def rank_auc(y, s):  # AUC exacto (Mann-Whitney, maneja empates)
    r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)

def metrics_at(y, pred):
    tp = int(((pred==1)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    acc = (tp+tn)/len(y)
    def f1(p, r): return 0 if p+r==0 else 2*p*r/(p+r)
    f1_pos = f1(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0)
    f1_neg = f1(tn/(tn+fn) if tn+fn else 0, tn/(tn+fp) if tn+fp else 0)
    den = np.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = 0 if den==0 else (tp*tn-fp*fn)/den
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, accuracy=acc, f1_macro=(f1_pos+f1_neg)/2, mcc=mcc)

y = np.load(M/"qsvm_y_test.npy")
print(f"y_test: n={len(y)}  positivos={int(y.sum())} ({y.mean():.4f})  negativos={int((1-y).sum())}\n")

for name, prefix in [("lgbm","lgbm"), ("svm","svm"), ("qsvm","qsvm")]:
    s = np.load(M/f"{prefix}_y_scores.npy")
    d = DECL[name]
    print(f"=== {name.upper()} ===  scores: min={s.min():.4f} max={s.max():.4f} n={len(s)}")
    a_trap, a_rank = auc_of(y, s), rank_auc(y, s)
    print(f"  AUC declarado={d['auc']:.4f} | recalculado(trapecio)={a_trap:.4f} | exacto(Mann-Whitney)={a_rank:.4f}"
          f"  {'OK' if abs(a_rank-d['auc'])<0.002 else '<<< DISCREPANCIA'}")
    m50 = metrics_at(y, (s>=0.5).astype(int))
    print(f"  CM declarada       tn={d['tn']} fp={d['fp']} fn={d['fn']} tp={d['tp']}"
          f"  acc={d['accuracy']:.4f} f1M={d['f1_macro']:.4f} mcc={d['mcc']:.4f}")
    print(f"  CM @umbral 0.50    tn={m50['tn']} fp={m50['fp']} fn={m50['fn']} tp={m50['tp']}"
          f"  acc={m50['accuracy']:.4f} f1M={m50['f1_macro']:.4f} mcc={m50['mcc']:.4f}"
          f"  {'OK' if (m50['tp'],m50['fp'])==(d['tp'],d['fp']) else '<<< NO COINCIDE CON 0.5'}")
    # buscar el umbral que reproduce la CM declarada.
    # OJO: hay que incluir 0.0 y los puntos MEDIOS entre scores consecutivos, no solo los
    # valores de score observados. El QSVM corta en decision_function > 0 y ningun score
    # vale exactamente 0: barrer solo los valores existentes daba un falso "NINGUNO".
    u = np.unique(np.round(s, 6))
    cands = np.unique(np.concatenate([[0.0], u, (u[:-1] + u[1:]) / 2.0]))
    best = None
    for th in cands:
        mm = metrics_at(y, (s >= th).astype(int))
        if (mm['tp'], mm['fp'], mm['tn'], mm['fn']) == (d['tp'], d['fp'], d['tn'], d['fn']):
            best = th; break
    print(f"  umbral que reproduce la CM declarada: {best if best is not None else 'NINGUNO'}")
    print()
