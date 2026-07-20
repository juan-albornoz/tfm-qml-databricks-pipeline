# Exportar los scores reales para las curvas ROC de LightGBM y SVM-RBF

Objetivo: generar 4 archivos `.npy` (scores + etiquetas de cada modelo) para que el
dashboard de Streamlit dibuje las curvas ROC de LightGBM y SVM-RBF con los datos
**reales**, igual que ya hace con el QSVM. Es una celda nueva en cada notebook.

Los vectores de scores ya están calculados en memoria en ambos notebooks — solo hay
que guardarlos con `np.save`. Se exportan también las etiquetas de cada modelo para
que `scores[i]` y `label[i]` queden siempre alineados (no se asume que el orden del
test sea idéntico entre notebooks).

---

## NB 04 — `notebook_04_lgbm.ipynb`  (LightGBM)

La celda 17 ya calcula `y_pred_proba` (probabilidad de clase 1) y `y_test`.
**Añade una celda nueva justo después de la celda 17** (o en cualquier punto tras
ella, antes de que esas variables se reasignen) con esto:

```python
# Persistir scores + etiquetas reales del test para la curva ROC del dashboard
import numpy as np
np.save(f"{models_dir}/lgbm_y_scores.npy", np.asarray(y_pred_proba))
np.save(f"{models_dir}/lgbm_y_test.npy",   np.asarray(y_test))
print(f"✅ Scores LightGBM persistidos: {len(y_pred_proba)} instancias")
print(f"   AUC de control: {roc_auc_score(y_test, y_pred_proba):.4f}")   # debe dar 0.9485
```

---

## NB 05 — `notebook_05_svm.ipynb`  (SVM-RBF)

La celda 13 ya calcula `y_prob` (probabilidad de clase 1) y `y_test`.
**Añade una celda nueva justo después de la celda 13** con esto:

```python
# Persistir scores + etiquetas reales del test para la curva ROC del dashboard
import numpy as np
np.save(f"{models_dir}/svm_y_scores.npy", np.asarray(y_prob))
np.save(f"{models_dir}/svm_y_test.npy",   np.asarray(y_test))
print(f"✅ Scores SVM-RBF persistidos: {len(y_prob)} instancias")
print(f"   AUC de control: {roc_auc_score(y_test, y_prob):.4f}")        # debe dar 0.9377
```

---

## Después de ejecutar

`models_dir` en los notebooks apunta a `/Volumes/.../nhanes/models` (Databricks).
Descarga los 4 archivos y colócalos en **`streamlit/models/`** del repo, junto a los
del QSVM que ya están ahí:

```
streamlit/models/lgbm_y_scores.npy
streamlit/models/lgbm_y_test.npy
streamlit/models/svm_y_scores.npy
streamlit/models/svm_y_test.npy
```

Comprobación rápida de que quedaron bien (deben coincidir con el AUC del TFM):
el de LightGBM ~0.9485 y el de SVM-RBF ~0.9377. Si dan eso, Claude conecta las
tres curvas ROC a datos reales en `app.py`.
```
