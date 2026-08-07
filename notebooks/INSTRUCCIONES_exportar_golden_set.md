# Exportar el *golden set* para la verificación end-to-end del dashboard

**Objetivo.** Hoy el dashboard puede comprobar que las métricas publicadas cuadran con los
scores guardados (`reconciliar_metricas()`), pero **no** puede comprobar que su propio camino
de inferencia —construir el vector, escalar solo el SVM, llamar al ONNX, leer el tensor
correcto— reproduce lo que produjeron los modelos entrenados. Falta el dato para hacerlo: el
conjunto de test no está en el repositorio.

El *golden set* lo cierra: **25 filas del test con la probabilidad que devolvió el modelo
entrenado**. Con eso el dashboard verifica de punta a punta y lo dice en pantalla.

Son dos celdas nuevas, una por notebook. No modifican nada existente.

---

## Por qué las dos celdas no son simétricas

Es el detalle que hay que respetar, y viene del NB03 (celda 26):

```python
X_train_lgbm.to_parquet(f"{gold_dir}/X_train_lgbm.parquet", index=False)   # CRUDO
X_train_svm_scaled.to_parquet(f"{gold_dir}/X_train_svm.parquet", index=False)   # ESCALADO
```

El fichero `X_train_svm.parquet` **contiene los datos escalados** pese a que el nombre no lo
diga. Por eso LightGBM se entrena con datos crudos y el SVM con datos escalados, y por eso el
dashboard escala **solo** para el SVM.

El golden set debe guardar siempre valores **CRUDOS**, para que el dashboard tenga que aplicar
su propio escalado y ese paso quede verificado. En el NB04 ya lo son; en el NB05 hay que
invertir el escalado con el propio `scaler_correcto.json`.

---

## NB 04 — `notebook_04_lgbm.ipynb` (LightGBM)

Añade una celda **después de la celda 15** (tras `lgbm_final.fit(...)`), en cualquier punto en
que `lgbm_final`, `X_test` e `y_test` sigan vivos.

```python
# ── Golden set para la verificación end-to-end del dashboard ────────────────────
# 25 filas del test + la probabilidad que produce el modelo entrenado. El dashboard
# las pasa por SU camino de inferencia y comprueba que obtiene lo mismo.
# X_test del NB04 es CRUDO (sin escalar): LightGBM se entrena así. Se guarda tal cual,
# incluidos los NaN — LightGBM los maneja de forma nativa y hay que preservarlos.
import numpy as np

K = 25
proba = lgbm_final.predict_proba(X_test)[:, 1]
y_arr = np.asarray(y_test)
rng = np.random.default_rng(42)

# Muestra deliberada: positivos, negativos, los dos extremos del score y un barrido
# por cuantiles. Así el golden set cubre todo el rango de salida y no solo el centro.
idx_pos, idx_neg = np.where(y_arr == 1)[0], np.where(y_arr == 0)[0]
sel = np.concatenate([
    rng.choice(idx_pos, 8, replace=False),
    rng.choice(idx_neg, 8, replace=False),
    [int(np.argmin(proba)), int(np.argmax(proba))],
    np.argsort(proba)[np.linspace(0, len(proba) - 1, 9).astype(int)],
])
sel = np.unique(sel)[:K]

np.savez(
    f"{models_dir}/golden_lgbm.npz",
    X=X_test.iloc[sel].to_numpy(dtype=np.float64),      # CRUDO, con NaN si los hay
    p=proba[sel].astype(np.float64),                    # referencia: predict_proba del PKL
    y=y_arr[sel].astype(np.int64),
    features=np.array(list(X_test.columns), dtype=object),
    escalado=np.array(False),                           # el dashboard NO debe escalar
)

print(f"golden_lgbm.npz — {len(sel)} filas")
print(f"  rango de probabilidad: {proba[sel].min():.4f} … {proba[sel].max():.4f}")
print(f"  positivos: {int(y_arr[sel].sum())} / {len(sel)}")
print(f"  NaN en la muestra: {int(np.isnan(X_test.iloc[sel].to_numpy(dtype=np.float64)).sum())}")
```

---

## NB 05 — `notebook_05_svm.ipynb` (SVM-RBF)

Añade una celda **después de la celda 11** (tras `svm_model.fit(...)`).

```python
# ── Golden set para la verificación end-to-end del dashboard ────────────────────
# OJO: el X_test de este notebook ya viene ESCALADO (el NB03 guardó X_test_svm_scaled
# bajo el nombre X_test_svm.parquet). El golden set tiene que llevar valores CRUDOS
# para que sea el DASHBOARD quien aplique el escalado — que es justo el paso que
# queremos verificar. Se invierte con el mismo scaler que consume la app.
import numpy as np, json

sc = json.load(open(f"{models_dir}/scaler_correcto.json"))
assert list(X_test.columns) == sc["features"], "el orden de features no coincide con el scaler"
mean = np.asarray(sc["mean"], dtype=np.float64)
scale = np.asarray(sc["scale"], dtype=np.float64)

X_scaled = X_test.to_numpy(dtype=np.float64)
X_raw = X_scaled * scale + mean                      # deshacer (x - mean) / scale

# Comprobación del viaje de ida y vuelta antes de guardar nada.
err = np.abs((X_raw - mean) / scale - X_scaled).max()
assert err < 1e-9, f"la inversión del escalado no es exacta: {err:.2e}"

K = 25
proba = svm_model.predict_proba(X_test)[:, 1]        # sobre los datos ESCALADOS
y_arr = np.asarray(y_test)
rng = np.random.default_rng(42)

idx_pos, idx_neg = np.where(y_arr == 1)[0], np.where(y_arr == 0)[0]
sel = np.concatenate([
    rng.choice(idx_pos, 8, replace=False),
    rng.choice(idx_neg, 8, replace=False),
    [int(np.argmin(proba)), int(np.argmax(proba))],
    np.argsort(proba)[np.linspace(0, len(proba) - 1, 9).astype(int)],
])
sel = np.unique(sel)[:K]

np.savez(
    f"{models_dir}/golden_svm.npz",
    X=X_raw[sel],                                    # CRUDO: el dashboard lo escalará
    p=proba[sel].astype(np.float64),
    y=y_arr[sel].astype(np.int64),
    features=np.array(list(X_test.columns), dtype=object),
    escalado=np.array(True),                         # el dashboard SÍ debe escalar
)

print(f"golden_svm.npz — {len(sel)} filas")
print(f"  rango de probabilidad: {proba[sel].min():.4f} … {proba[sel].max():.4f}")
print(f"  positivos: {int(y_arr[sel].sum())} / {len(sel)}")
print(f"  error maximo del round-trip de escalado: {err:.2e}")
```

---

## Desplegar

Copia los dos ficheros del volumen a `streamlit/models/`:

```
golden_lgbm.npz
golden_svm.npz
```

El `.gitignore` ya tiene la excepción `!streamlit/models/golden_*.npz`, así que suben al
repositorio como los `.onnx` y los `.npy`.

Si no están, el dashboard no falla: la tarjeta de verificación aparece como *no disponible*
con esta misma instrucción. Nunca afirma haber verificado algo que no ha podido comprobar.

---

## Qué verifica exactamente

Con los ficheros presentes, `verificar_golden()` (en `app.py`) reconstruye para cada fila el
camino completo del dashboard y lo compara con `p`:

| Paso | LightGBM | SVM-RBF |
|-|-|-|
| Entrada | cruda, tal cual | cruda |
| Escalado `(x − mean) / scale` | **no** | **sí** |
| Conversión a `float32` | sí | sí |
| Sesión ONNX | `lgbm_final.onnx` | `svm_final.onnx` |
| Lectura del tensor | `out[1]` es `ndarray (N,2)` → `[i][1]` | `out[1]` es lista de `dict` → `[i][1]` |

Ese último paso no es un detalle: **cada modelo devuelve una estructura distinta** porque el
NB04 convirtió con `zipmap: False` y el NB05 sin esa opción. La expresión `out[1][i][1]`
funciona en los dos, pero por motivos diferentes, y una regresión ahí daría números
plausibles pero falsos. El golden set lo detecta.

**Tolerancia:** 1 × 10⁻⁴ en probabilidad absoluta. Los `.onnx` operan en `float32` y los
`predict_proba` de referencia son `float64`, así que se esperan diferencias del orden de
10⁻⁷; una discrepancia mayor indica un problema real de pipeline, no ruido numérico.
