# Informe de auditoría — Dashboard Streamlit (comportamiento numérico)

**Proyecto:** `tfm-qml-databricks-pipeline` · `streamlit/app.py` (2.665 líneas)
**Fecha:** 6 de agosto de 2026
**Alcance:** las 6 páginas del dashboard y sus pestañas internas, con verificación **dinámica**: los modelos ONNX se ejecutaron realmente y las métricas declaradas se recalcularon contra los `.npy` del test.

> Complementa a [INFORME\_AUDITORIA\_TFM.md](INFORME_AUDITORIA_TFM.md), que fue análisis **estático** de notebooks. Aquí sí se ejecutó código: inferencia ONNX real sobre `lgbm_final.onnx` y `svm_final.onnx`, y recálculo de AUC / matrices de confusión / F1-macro / MCC sobre `qsvm_y_test.npy` y los tres ficheros de scores.
>
> Scripts reproducibles en [`audit_scripts/`](audit_scripts/). Se ejecutan sin argumentos desde cualquier directorio: resuelven `streamlit/models/` de forma relativa a su propia ubicación.

---

## 0. Resumen ejecutivo

| Severidad | Nº | Dónde |
|-|-|-|
| Crítico | 4 | Predictor en Vivo (4) |
| Alto | 7 | Predictor en Vivo (2), Esfera de Bloch (2), Resultados (1), Gobernanza (1), Circuito (1) |
| Medio | 8 | Predictor en Vivo (3), Resultados (1), Gobernanza (1), SHAP (1), Circuito (1), Bloch (1) |
| Bajo / cosmético | 5 | varias |

**Lo que sí está bien — y es mucho:** los tres AUC (0,9485 / 0,9377 / 0,5493) reproducen **exactamente** con Mann-Whitney sobre los scores guardados. Las matrices de confusión de **los tres** modelos, con su accuracy, F1-macro y MCC, reproducen exactamente cada una en su umbral (LightGBM 0,50 · SVM signo de `decision_function` · QSVM `> 0`). El embudo de registros de Gobernanza cuadra aritméticamente escalón a escalón. Las especificaciones del circuito cuántico (8 qubits, reps=2, linear, muestra de 500, test de 1.567, support vectors [425, 70]) coinciden con `notebook_06`. El ranking RF de la página Circuito Cuántico coincide dígito a dígito con `notebook_03`.

Dicho de otro modo: **ninguna métrica de modelo está mal calculada**. Los problemas son de *encuadre* (qué se dice que significan los números) y de *contexto* (qué no se dice al ponerlos juntos), más un puñado de cifras narrativas sin respaldo.

**El titular:** tu intuición sobre HbA1c es correcta, y la causa no es un bug de la app — es **qué predice el modelo**. `TARGET = (DIQ010 == 1)`, es decir *"¿un médico le ha dicho alguna vez que tiene diabetes?"*. El modelo no estima riesgo de desarrollar diabetes: detecta **diabetes ya diagnosticada**. La HbA1c es el marcador con el que se hace y se sigue ese diagnóstico, así que el modelo la usa casi como si fuera la etiqueta. De ahí la pendiente.

---

## 1. PREDICTOR EN VIVO

### 1.1 Reproducción exacta de tu observación

Ejecutando `predict_real()` con el resto de sliders en sus valores por defecto:

| HbA1c | LightGBM | SVM-RBF |
|-|-|-|
| 5,8 % | **32,28 %** | 2,64 % |
| 6,3 % | **88,07 %** | 11,48 % |

Tus cifras (32,3 → 88,1) son exactas. **+55,8 puntos en 0,5 unidades de HbA1c.**

### 1.2 [CRÍTICO] El modelo no predice riesgo — detecta un diagnóstico existente

`notebook_02_silver.ipynb`:

```python
df_silver["TARGET"] = np.where(df_silver["DIQ010"] == 1.0, 1, 0)
```

DIQ010 es la pregunta de autoinforme *"Doctor told you have diabetes"*. Consecuencias directas, todas medidas:

- **La HbA1c es consecuencia del target, no su predictor.** Un diagnosticado tiene HbA1c alta casi por definición. El AUC de 0,9485 mide capacidad de **detección concurrente**, no de predicción prospectiva.
- **Confusión por tratamiento en LDL — el signo sale invertido.** Barrido aislado: LDL 40 mg/dL → 43,1 % ; LDL 250 mg/dL → **17,5 %**. Más colesterol LDL, *menos* riesgo predicho. Es coherente con que los diabéticos diagnosticados están estatinizados, pero la página presenta LDL como factor de riesgo sin advertirlo.
- **Glucosa con forma en U.** 50 mg/dL → 44,0 % ; 112 mg/dL → 38,3 % ; 175 mg/dL → 84,5 %. La hipoglucemia sube el riesgo predicho (diabéticos tratados).
- **La insulina es casi inerte.** Todo su rango (2–60 µU/mL) mueve la predicción de 0,313 a 0,349: **3,6 puntos**, pese a ser la 6ª feature del ranking RF.

**Recomendación para la defensa:** renombrar la página y la métrica. No es "riesgo de diabetes" sino *"probabilidad de que este perfil corresponda a una persona ya diagnosticada"*. Es un resultado perfectamente defendible — pero solo si se enuncia así.

### 1.3 [ALTO] El 84,5 % del slider de HbA1c es zona muerta

Rango del slider: 4,0 – 15,0 %. Respuesta real de LightGBM:

| Tramo | Anchura | % del slider | Comportamiento |
|-|-|-|-|
| 4,0 – 5,55 | 1,55 | 14,5 % | **plano en 24,29 %** |
| 5,55 – 7,30 | 1,75 | 15,5 % | zona activa (14 escalones) |
| 7,30 – 15,0 | 7,70 | **70,0 %** | **plano en 97,31 %** |

La causa está en la winsorización IQR × 3 de Silver. El scaler recuperado da para LBXGH media = 5,72 y **sd = 0,77** → ±3 sd = [3,41 ; 8,03]. El slider llega a 15,0, que es **z = +12,04**: el modelo nunca vio nada parecido. Toda la potencia discriminante quedó comprimida en una banda de 1,75 puntos.

Mismo problema en el resto de sliders:

| Variable | Slider | media ± sd (train) | z(mín) | z(máx) |
|-|-|-|-|-|
| LBXGH | 4,0 – 15,0 | 5,72 ± 0,77 | −2,23 | **+12,04** |
| LBXGLU | 50 – 300 | 107,03 ± 21,26 | −2,68 | **+9,07** |
| LBXIN | 2 – 60 | 12,56 ± 9,71 | **−1,09** | +4,89 |
| LBDLDL | 40 – 250 | 110,16 ± 34,53 | −2,03 | +4,05 |
| BMXBMI | 15 – 60 | 29,20 ± 7,11 | −2,00 | +4,33 |
| RIDAGEYR | 18 – 80 | 48,93 ± 18,44 | −1,68 | +1,68 |

Solo la edad está bien acotada. En insulina el mínimo del slider queda a **1,09 sd** de la media: el usuario no puede explorar hipoinsulinemia.

**Recomendación:** recortar los sliders al rango observado en entrenamiento (p. ej. HbA1c 4,0–9,0) o marcar visualmente la frontera de extrapolación.

### 1.4-bis Por qué los peldaños caen siempre entre dos posiciones del slider

Localizando los umbrales por bisección sobre el propio ONNX (precisión 10⁻⁶), los **14 cortes de LBXGH caen todos exactamente en X,X5**:

```
5,55  5,65  5,75  5,85  6,05  6,15  6,25  6,35  6,45  6,65  6,75  6,85  6,95  7,25
```

No es casualidad. **NHANES registra la HbA1c con un decimal**, y LightGBM sitúa cada split en el punto medio entre dos valores observados consecutivos: entre 6,0 y 6,1 el corte es 6,05. Consecuencias:

- El slider (paso 0,1) **nunca puede posarse sobre un umbral**: cae siempre a un lado u otro, así que el usuario solo ve saltos, nunca la transición.
- El modelo **no tiene resolución por debajo de 0,1 % de HbA1c**. Pedirle un valor intermedio no tiene sentido: los datos no lo soportan.
- De las 111 posiciones del slider salen **solo 15 probabilidades distintas**.
- Tres peldaños (6,05 · 6,15 · 6,25) concentran el **66 % de todo el recorrido** de la variable.

**Caso reportado por el autor (6 ago 2026):** «6,0 → 27,1 %, pero 6,1 → 43,9 %». Reproducido exactamente — corresponde a `BMXWAIST ≤ 80 cm` con el resto por defecto. Es el peldaño de 6,05 visto desde una base más baja, no un fallo. Descartada también deriva de coma flotante: valores como 6,0499999 y 6,0500001 caen ambos del mismo lado, y el salto está limpio entre 6,05 y 6,06.

**Corregido en la app** añadiendo una *curva de respuesta* (§8, grupo E): un gráfico de la probabilidad frente a la variable elegida, dibujado con `line_shape="hv"` —escalón literal, nunca suavizado— con el punto actual marcado, las bandas ADA y un pie que cuenta los peldaños reales de la configuración vigente. El salto deja de ser una sorpresa y pasa a ser una propiedad visible del modelo.

### 1.4 [ALTO] Escalones discretos y cruce del umbral en zona de prediabetes

LightGBM es un ensemble de árboles: la respuesta es una **función escalón**, no una curva. Los 14 escalones localizados con barrido de 0,01:

| Umbral del árbol | Antes → después | Salto |
|-|-|-|
| 5,56 | 0,3127 → 0,3228 | +0,0101 |
| **5,86** | 0,3228 → 0,4425 | **+0,1197** |
| **6,06** | 0,4425 → 0,6258 | **+0,1833** ← cruza el 50 % |
| 6,16 | 0,6258 → 0,7422 | +0,1164 |
| 6,26 | 0,7422 → 0,8807 | +0,1385 |

**El paso a "Riesgo Elevado" ocurre en HbA1c = 6,06 %.** Los criterios ADA sitúan prediabetes en 5,7–6,4 % y diabetes en ≥ 6,5 %. El modelo declara "Elevado" en pleno rango de prediabetes, medio punto antes del criterio diagnóstico. En una defensa, un tribunal clínico lo va a señalar.

**Recomendación:** superponer al velocímetro las bandas ADA (< 5,7 / 5,7–6,4 / ≥ 6,5) y explicar en la nota que un ensemble de árboles produce escalones, no una rampa suave.

### 1.5 [CRÍTICO] La app llama "sustituto" a la inferencia real

`app.py:2590` y `app.py:2578`, en el bloque que se pinta **siempre**:

```python
<div class="stat-label">Score de riesgo (sustituto)</div>
...
"El score supera el umbral de decisión (50%): el sustituto clasificaría como caso positivo."
```

Los modelos ONNX **sí están cargados** (verificado: `lgbm_final.onnx` y `svm_final.onnx` presentes y funcionales), así que arriba se muestra el aviso *"Inferencia real (ONNX)"* y abajo la cifra se rotula como maqueta. Se contradicen en la misma pantalla. Hay que condicionar ambos textos a `_real is not None`.

### 1.6 [CRÍTICO] El SVM se calcula y se descarga

```python
risk, _svm_prob = _real   # app.py:2554
```

`_svm_prob` no se usa en ninguna parte. El encabezado de la página promete *"LightGBM y SVM-RBF"* y solo se muestra LightGBM.

### 1.7 [CRÍTICO] Si se mostrara el SVM, el umbral 0,5 sería erróneo

En `notebook_05_svm.ipynb` las métricas salen de `svm_model.predict(X_test)`. En scikit-learn, `SVC(probability=True).predict()` usa el **signo de `decision_function`**, no `predict_proba() >= 0.5` — son inconsistentes por diseño. Verificado numéricamente:

| | tn | fp | fn | tp | accuracy |
|-|-|-|-|-|-|
| CM declarada en la app | 1250 | 97 | 48 | 172 | 0,9075 |
| Recalculada a umbral 0,50 | 1309 | 38 | 89 | 131 | 0,9190 |
| **Umbral que reproduce la declarada** | | | | | **0,2217** |

El punto de operación real del SVM está en **p ≈ 0,22**, no en 0,50. Las bandas de la página (< 0,33 Bajo / < 0,50 Moderado / ≥ 0,50 Elevado) no le aplican.

### 1.8 [MEDIO] Los dos modelos se contradicen justo donde importa

| Perfil | LightGBM | SVM-RBF | Diferencia |
|-|-|-|-|
| Sano joven (HbA1c 5,0) | 6,3 % | 0,1 % | 6 pts |
| **Prediabético (HbA1c 6,0)** | **64,7 %** | **8,8 %** | **56 pts** |
| Diabético franco (HbA1c 8,5) | 99,2 % | 100,0 % | 1 pt |
| Descontrolado (HbA1c 11,0) | 99,3 % | 98,3 % | 1 pt |

Coinciden en los extremos y divergen brutalmente en la zona de decisión clínica. Presentar solo LightGBM oculta esa incertidumbre.

### 1.9 [MEDIO] El paciente base es imposible

`_build_feature_vector()` fija las 81 variables no editables a la **mediana por columna**. Para las columnas one-hot eso da suma cero:

| Grupo | Suma en el vector base | Moda real (prevalencia) |
|-|-|-|
| RIAGENDR | **0** | RIAGENDR_1.0 (0,485) |
| RIDRETH1 | **0** | RIDRETH1_3.0 (0,373) |
| RIDRETH3 | **0** | RIDRETH3_6.0 (0,128) |
| DMDEDUC2 | **0** | DMDEDUC2_4.0 (0,338) |
| DMDMARTL | 1 | ✓ correcto |

El paciente de referencia no tiene sexo, ni etnia, ni nivel educativo. Impacto medido al corregirlo con la moda: LightGBM **±0,0000** (no ramifica sobre esos dummies), SVM **+0,0108** (de 0,0194 a 0,0302, un +56 % relativo). Poco impacto práctico, pero es un error metodológico visible si alguien audita el código.

### 1.10 [MEDIO] El peso muestral mueve la predicción 9 puntos

WTINT2YR es el factor de expansión de la encuesta NHANES — la propia app lo describe como *"artefacto del diseño muestral, no una variable clínica"* y es la **6ª feature por SHAP**. Barrido:

| WTINT2YR | LightGBM |
|-|-|
| 5.000 | 33,71 % |
| 25.697 (mediana) | 31,27 % |
| 120.000 | **24,38 %** |

**9,3 puntos de "riesgo de diabetes" atribuibles a cómo se muestreó al participante.** Ya figura como hallazgo 09/10 en Gobernanza; aquí queda cuantificado.

### 1.11 [BAJO] Pesos del fallback obsoletos

Las importancias de `QSVM_FEATURES` (0,2452 / 0,1853 / 0,0325 …) no coinciden con las del `notebook_03` ni con `RF_TOP8_IMPORTANCE` de la propia app (0,2454 / 0,1855 / 0,0323 …). El comentario del código dice que la divergencia es deliberada, pero el resultado es que la misma magnitud aparece con dos valores distintos en la misma aplicación. Solo afecta al score sustituto (inactivo).

---

## 2. RESULTADOS

### 2.1 Verificado ✓

Recalculado con Mann-Whitney sobre los `.npy` (n = 1.567; 220 positivos, 14,04 %):

| Modelo | AUC declarado | AUC recalculado | |
|-|-|-|-|
| LightGBM | 0,9485 | 0,9485 | ✓ |
| SVM-RBF | 0,9377 | 0,9377 | ✓ |
| QSVM | 0,5493 | 0,5493 | ✓ |

LightGBM: matriz de confusión (924/423/9/211), accuracy 0,7243, F1-macro 0,6523 y MCC 0,4566 reproducen **exactamente** al umbral 0,50. ✓

### 2.2 La CM del QSVM es correcta ✓

La app declara tn=1347, fp=0, fn=219, tp=1, accuracy 0,8602, F1-macro 0,4669, MCC 0,0625. **Todo reproduce exactamente** aplicando `y_scores > 0`, que es el umbral que usa `notebook_06_qsvm.ipynb` (celda 23) y el umbral natural de un SVM sobre `decision_function`:

```
s > 0     tn=1347 fp=0 fn=219 tp=1   acc=0.8602 f1M=0.4669 mcc=0.0625
declarado tn=1347 fp=0 fn=219 tp=1   acc=0.8602 f1M=0.4669 mcc=0.0625
```

Solo una instancia del test (índice 130, etiqueta real = 1) supera el umbral, de ahí el tp=1.

> **Nota de corrección.** Una versión previa de este informe daba esta matriz por irreproducible. Era un fallo del script de auditoría: `audit_metrics.py` barría como umbrales candidatos únicamente los valores de score existentes, y ninguno es exactamente 0, así que el umbral correcto nunca se probó. La app está bien; el script estaba mal. Corregido en `audit_scripts/audit_metrics.py`.

### 2.3 [ALTO] Los tres modelos se comparan en puntos de operación distintos

Cada modelo está reportado en **su** umbral natural, y cada uno por separado es correcto:

| Modelo | Cómo se obtuvo `y_pred` | Umbral en escala de probabilidad |
|-|-|-|
| LightGBM | `predict_proba()[:,1] >= 0.5` | 0,50 |
| SVM-RBF | `svm_model.predict()` → signo de `decision_function` | **≈ 0,2217** |
| QSVM | `y_scores > 0` → `decision_function` | no es probabilidad |

El problema no es ningún modelo aislado, sino **ponerlos lado a lado**: las tarjetas KPI, las tres matrices de confusión y el gráfico de barras invitan a leer accuracy, F1-macro y MCC como comparables, y no lo son — dependen del umbral, y aquí hay tres umbrales distintos. Los AUC sí son comparables (son independientes del umbral).

Ejemplo del efecto: al SVM, evaluado a 0,50 como LightGBM, le saldría accuracy 0,9190 (mejor) pero solo 131 verdaderos positivos en vez de 172 (peor recall). La comparación cambia de sentido según dónde se corte.

### 2.4 [MEDIO] Métricas hardcodeadas teniendo los scores en disco

`MODELS` (app.py:1002-1007) lleva las métricas transcritas a mano, aunque los tres ficheros de scores están embarcados y permiten recalcularlas en caliente. La desincronización del QSVM (2.2) es precisamente lo que este patrón provoca.

### 2.5 [BAJO] Curvas ROC empíricas suavizadas

`shape="spline", smoothing=0.4` sobre una curva ROC empírica, que por construcción es escalonada. Cosmético, pero se presenta como "curva empírica real, punto a punto".

---

## 3. GOBERNANZA

### 3.1 Verificado ✓

Embudo aritméticamente consistente: 29.400 − 11.439 = 17.961 − 10.126 = 7.835 − 4 = **7.831** ✓. Partición 6.264 + 1.567 = 7.831 ✓. Prevalencia 14,03 % train / 14,04 % test ✓ (220/1.567 confirmado en el `.npy`).

### 3.2 [ALTO] Hallazgo no listado: 23 de las 89 features tienen varianza cero

El `scaler_correcto.json` tiene `scale == 1.0` exacto en 23 columnas, marca inequívoca de varianza nula. El propio `notebook_03` lo imprime (*"Columnas excluidas (varianza = 0): 23"*), pero el dashboard no lo menciona en ningún sitio:

```
DMQMILIZ, DMDCITZN, SIALANG, SIAPROXY, SIAINTRP, FIALANG, FIAPROXY,
FIAINTRP, MIALANG, MIAPROXY, MIAINTRP, AIALANGA, DMDHHSZA, BMDSTATS,
BPAARM, BPXPULS, BPXPTY, BPAEN1, BPAEN2, BPAEN3, PAQ605, PAQ635, PAQ650
```

**El 25,8 % de las "89 features" anunciadas es constante y no aporta nada.** Es la consecuencia directa del problema ya documentado en `TECHNICAL_NOTES.md` línea 141: la winsorización IQR × 3 se aplicó a variables categóricas codificadas 1/2, y cuando más del 75 % responde lo mismo, el recorte las colapsa a constante. El informe estático lo documentó como riesgo; aquí queda confirmado el daño exacto.

Se puede convertir en un punto fuerte de la memoria: *"de 89 features nominales, 66 son efectivas"*, con la lista y la causa.

### 3.3 [MEDIO] Off-by-one en "Operaciones de calidad por capa"

La tarjeta Gold encadena: 106 tras codificación → −16 por correlación → **89 finales**. Pero 106 − 16 = 90. El `notebook_03` aclara el desajuste: los recuentos de 106 y 90 son **columnas del DataFrame e incluyen TARGET**; las features son 105 y 89. La tarjeta mezcla los dos criterios de conteo.

### 3.4 [BAJO] "15/15 artefactos sin leakage" convive con el hallazgo 09

El KPI es rotundo; el hallazgo 09 de la misma página admite que el filtro de correlación *"se calcula sobre el dataset completo, así que las 16 columnas descartadas se deciden usando también las observaciones de test"*. No es contradictorio si "artefactos" se refiere a ficheros, pero el titular promete más de lo que el matiz sostiene. Sugerencia: *"15/15 artefactos sin fuga de etiqueta"*, que es lo que realmente se comprobó.

---

## 4. ANÁLISIS SHAP

### 4.1 [MEDIO] "El ranking del SVM coincide con LightGBM" — solo en composición

La nota afirma coincidencia en las variables dominantes. Cierto en el conjunto, no en el orden:

| Puesto | LightGBM | SVM-RBF |
|-|-|-|
| 1 | LBXGH (1,1243) | LBXGH (0,1017) |
| 2 | **RIDAGEYR (0,4654)** | LBXGLU (0,0436) |
| 3 | LBXGLU (0,3161) | LBDLDL (0,0219) |
| 4 | LBDLDL (0,2542) | **RIDAGEYR (0,0141)** |

La edad es 2ª en LightGBM y 4ª en el SVM. Además las escalas difieren en un orden de magnitud (1,12 vs 0,10) y **no son comparables entre sí**: TreeExplainer devuelve valores exactos en log-odds sobre las 1.567 instancias; KernelExplainer aproxima por muestreo (fondo de 100, contribuciones sobre 200). Las notas al pie de cada pestaña sí lo dicen; la afirmación de "robustez metodológica" del texto debería matizarlo.

### 4.2 [BAJO] Claves de tooltip sin sufijo

`VAR_DESC` usa `DMDMARTL_1`, `DMDEDUC2_3`; el vector real usa `DMDMARTL_1.0`, `DMDEDUC2_3.0`. Hoy solo afecta a tooltips (coinciden con `SHAP_SVMRBF`, que usa la misma forma corta). Pero `_build_feature_vector` descarta en silencio cualquier clave desconocida:

```python
for k, v in overrides.items():
    if k in feats:          # <- un nombre mal escrito se ignora sin avisar
        x[feats.index(k)] = v
```

Si alguna vez se amplía el Predictor con esas variables, el override no tendrá efecto y no habrá error. Merece un `else: raise` o al menos un aviso.

---

## 5. CIRCUITO CUÁNTICO

### 5.1 Verificado ✓

8 qubits, reps=2, entanglement linear, 8 parámetros, Qiskit 2.5.0, muestra de entrenamiento 500 (vía `train_test_split(train_size=500)` desde las 1.500 del Gold), test 1.567, `StatevectorSampler`. Todo coincide con `notebook_06_qsvm.ipynb` ✓. `RF_TOP8_IMPORTANCE` coincide dígito a dígito con la salida de la celda 20 del `notebook_03` ✓.

### 5.2 [ALTO] El tiempo de inferencia declarado no coincide con el notebook

La app declara **132,8 min** de tiempo de inferencia del QSVM en dos sitios: la tarjeta KPI de esta página (`app.py:2364`) y la nota al pie (`app.py:2377`), además de repetirlo en el aviso del Predictor en Vivo (`app.py:2525`). La salida real del `notebook_06_qsvm.ipynb`, celda 22:

```
Prediccion completada en 144.5 minutos
```

**La cifra 132,8 no aparece en ninguna salida del notebook.** Son 11,7 minutos de diferencia (−8,1 %). Hay que reemplazar los tres sitios por 144,5 min, o localizar de dónde salió 132,8 si corresponde a otra ejecución.

Adicionalmente, el **tiempo de entrenamiento** tiene dos valores dentro del propio notebook: la celda 19 imprime `Entrenamiento completado en 21.1 minutos`, pero los metadatos persistidos en la celda 30 registran `train_time_min: 22.6`.

> **Resuelto (6 ago 2026).** El autor confirma que el tiempo real fue **21,1 min**, que es el valor que ya mostraba la app (`app.py:2362`): **no requiere cambio**. El `22.6` de los metadatos de la celda 30 es el dato espurio.
>
> No se edita el notebook: sus outputs son un registro de ejecución y [INFORME\_AUDITORIA\_TFM.md](INFORME_AUDITORIA_TFM.md) los da por preservados byte a byte. La discrepancia queda documentada aquí para que una futura revisión no la reabra ni "corrija" la app hacia el valor equivocado.

Verificado correcto en cambio: `Support vectors: [425, 70]` ✓ coincide con la tarjeta.

### 5.3 [MEDIO] Contradicción interna: RZ vs P

- Cuerpo del texto: *"codifica cada una de las 8 variables clínicas como un ángulo de rotación **(puerta RZ)**"*
- Pie del diagrama, 90 líneas más abajo: *"codificación **(H + P)**"*

El `ZZFeatureMap` de Qiskit usa puertas **P (phase)**, que es lo que dicen el diagrama y el notebook. P y RZ difieren en una fase global, así que la afirmación no es falsa, pero es incoherente dentro de la misma página.

---

## 6. ESFERA DE BLOCH

### 6.1 [ALTO] Aliasing: el valor mínimo y el máximo dan el mismo estado

```python
theta = 2 * x_norm * np.pi     # app.py:2414
```

Con `x_norm ∈ [0,1]`, θ recorre **[0, 2π]** — una vuelta completa. Resultado:

| HbA1c | x_norm | θ | P(\|0⟩) |
|-|-|-|-|
| 4,0 % | 0,00 | 0 | **100 %** |
| 9,5 % | 0,50 | π | 0 % |
| 15,0 % | 1,00 | 2π | **100 %** |

**HbA1c 4,0 % y HbA1c 15,0 % producen un estado cuántico idéntico.** La parametrización estándar de la esfera de Bloch usa θ ∈ [0, π]; con [0, 2π] la representación es doblemente degenerada. Arreglo: `theta = x_norm * np.pi`.

### 6.2 [ALTO] α negativo sin explicación

`alpha = cos(theta/2)` es **negativo para todo x_norm > 0,5**. Con HbA1c = 12 % la tarjeta muestra α = −0,782 sin ninguna nota. Es matemáticamente correcto, pero al lado de "amplitud de |0⟩" y de una P(|0⟩) positiva resulta desconcertante. Se resuelve solo al corregir 6.1.

### 6.3 [MEDIO] La nota afirma más de lo que la visualización hace

> *"Este cálculo reproduce el primer paso de codificación del ZZFeatureMap real: θ = 2·x_norm·π"*

En el `ZZFeatureMap` el primer paso es **H seguido de P(2·xᵢ)**. Una puerta de fase tras una Hadamard deja el estado **en el ecuador** (θ = π/2 fijo, P(|0⟩) = P(|1⟩) = 50 % **siempre**) y codifica el dato en el ángulo **azimutal φ**. La visualización actual mueve el ángulo **polar** — es una codificación tipo RY — y hace variar P(|0⟩) de 0 a 100 %.

Además Qiskit aplica `2·xᵢ` sobre el dato escalado, no `2·x_norm·π` con normalización previa al rango fisiológico.

Como recurso didáctico está bien: transmite "el valor clínico se convierte en un ángulo". Pero la frase *"reproduce el primer paso de codificación del ZZFeatureMap real"* es más fuerte de lo que sostiene el código, y un tribunal con perfil de física cuántica lo puede preguntar. Dos salidas: (a) reformular como *"analogía del principio de codificación angular"*, o (b) implementar la codificación de fase real y mostrar el vector girando por el ecuador.

---

## 7. RESUMEN (página de inicio)

Todas las cifras verificadas ✓: 29.400 Bronze, 7.831 Silver, 89 features, 86 % / 14 %. El donut usa 86/14 redondeados en vez de 85,97/14,03 — irrelevante.

Nota positiva: el subtítulo dice *"Target binarizado: **1 = diabetes diagnosticada**"*, que es exacto. La imprecisión aparece solo en el Predictor en Vivo, donde lo mismo se llama "riesgo".

---

## 8. Plan de corrección

Ordenado por **cuánta falsedad elimina por unidad de trabajo**. La defensa del TFM ya se celebró: no hay fecha límite, el objetivo es que la app sea veraz y útil como producto.

### Grupo A — La app afirma algo que no es cierto ✅ APLICADO (6 ago 2026)

| # | Qué se corrigió | Estado |
|-|-|-|
| A1 | La inferencia ONNX real se rotulaba "sustituto" | ✅ hecho |
| A2 | "132,8 min" no existía en el notebook; son 144,5 | ✅ hecho |
| A3 | El Predictor llamaba "riesgo" a lo que es detección de diagnóstico | ✅ hecho |
| A4 | Tres modelos comparados en tres umbrales, sin decirlo | ✅ hecho |

Detalle de lo aplicado:

- **A1** — `_es_real`, `_quien` y `_etiqueta_score` condicionan ahora todos los textos al origen del número. Con ONNX cargado la tarjeta dice *"Probabilidad de diagnóstico existente"* y las interpretaciones hablan de *"el modelo"*; en fallback siguen diciendo *"sustituto"*.
- **A2** — 144,5 min en los tres sitios. Además se corrigió *"132,8 min **por instancia**"*, que era un error de magnitud: 144,5 min es el total de las 1.567 instancias, no el coste unitario.
- **A3** — nuevo bloque *"Qué estima este formulario"* con la definición de `TARGET = (DIQ010 == 1)`; subtítulo de página reescrito; bandas renombradas de *Riesgo Bajo/Moderado/Elevado* a *Compatibilidad Baja/Intermedia/Alta*; y nota de cierre *"Cómo leer estos resultados"* que advierte del signo invertido del LDL y de la forma en U de la glucosa.
- **A4** — nueva fila *Umbral* en las tres tarjetas KPI (con el origen en `title`); nota bajo las tarjetas explicando que solo el AUC-ROC es comparable; subtítulos de matrices de confusión y del gráfico de barras reescritos.

Verificado tras aplicar: `ast.parse` correcto, servidor recargado sin trazas de error, y sin residuos de los textos antiguos.

### Grupo B — La app calla algo que cambia la lectura ✅ APLICADO (6 ago 2026)

| # | Qué se añadió | Estado |
|-|-|-|
| B1 | Rango de entrenamiento bajo cada slider + aviso de extrapolación | ✅ hecho |
| B2 | Referencia ADA bajo el slider de HbA1c | ✅ hecho |
| B3 | El SVM mostrado junto al LightGBM, cada uno con su umbral | ✅ hecho |
| B4 | Las 23 features de varianza cero, en Gobernanza | ✅ hecho |
| B5 | Aviso del signo invertido de LDL y la U de glucosa | ✅ hecho (con A3) |

- **B1** — media y desviación se leen del `scaler_correcto.json`, no se transcriben. Cada slider muestra `media ± sd` y su intervalo ±3 sd; si el valor elegido se sale, aparece el z real y el aviso *"fuera del rango entrenado: el modelo extrapola"*. Se optó por **marcar** en vez de recortar: acotar los sliders habría exigido inventar límites que ningún artefacto respalda.
- **B2** — la banda ADA va bajo el slider de HbA1c, no sobre el velocímetro: el dial está en escala de **probabilidad**, no de HbA1c, así que superponerle cortes de 5,7/6,4/6,5 habría sido incorrecto (el informe original lo planteaba mal).
- **B3** — bloque *"Los dos modelos, lado a lado"* con ambas probabilidades, el umbral de cada una y cómo clasificaría cada modelo. Si difieren más de 25 puntos aparece un aviso de incertidumbre.
- **B4** — las constantes se **cuentan desde el scaler** (`scale_ == 1.0` marca varianza nula), no se escriben a mano: si el pipeline se reejecuta, la cifra se mueve sola. Incluye la lista completa y la explicación causal (winsorización IQR × 3 sobre categóricas), verificada contra el notebook 02: las variables más recortadas —PAQ635 (1.929), PAQ650 (1.847), PAQ605 (1.648), DMDHHSZA (1.535), DMDCITZN (1.150), SIALANG (897)— **son exactamente las que aparecen constantes**.

### Grupo C — Corrección técnica ✅ APLICADO (salvo C4)

| # | Qué se corrigió | Estado |
|-|-|-|
| C1 | `theta = x_norm * np.pi` (elimina el aliasing) | ✅ hecho |
| C2 | Nota del ZZFeatureMap reformulada (H+P codifica en φ) | ✅ hecho |
| C3 | Vector base con moda en los one-hot | ✅ hecho |
| C4 | Reexportar los `.onnx` con opset ≥ 7 | ⬜ pendiente — requiere reejecutar notebooks 04/05 en Databricks |

- **C3** verificado tras el cambio: los cinco grupos one-hot suman ahora 1, con `RIAGENDR_1.0`, `RIDRETH1_3.0`, `RIDRETH3_6.0`, `DMDEDUC2_4.0` y `DMDMARTL_1.0` activos. La predicción de LightGBM no se mueve (no ramifica sobre esas dummies); la del SVM sube de 1,94 % a 3,02 % en el perfil por defecto.

### Grupo E — Curva de respuesta ✅ APLICADO (6 ago 2026)

Añadido al Predictor en Vivo a raíz del caso 6,0 → 6,1 (§1.4-bis). Selector de cualquiera de las 8 variables; se recorre su rango completo dejando las otras siete en los valores actuales, y se dibujan **ambos modelos** con `line_shape="hv"`.

Decisión de diseño deliberada: **el escalón se dibuja como escalón**. Usar `shape="spline"` habría producido una curva bonita que sugiere una continuidad que el modelo no tiene — el mismo error cosmético que §2.5 señala en las curvas ROC.

El pie se calcula sobre la curva vigente, no está escrito a mano: número de valores distintos alcanzables y los tres mayores peldaños con su magnitud. Con la configuración del autor (cintura 80 cm) reporta: *«15 valores distintos en las 111 posiciones del slider · mayores peldaños: 6,3 (+23,7 %) · 6,1 (+16,8 %) · 6,2 (+14,4 %)»*.

Coste medido: barrido de 111 puntos × 2 modelos en una sola llamada ONNX por modelo, ~97 ms sin caché y prácticamente nada con ella (`st.cache_data`). Punto a punto costaba lo mismo por cada rerun del slider.

### Grupo D — Higiene ✅ APLICADO

- **`raise` en overrides desconocidos** (§4.2) — `_build_feature_vector` lanzaba silencio; ahora `KeyError`. Verificado: `DMDMARTL_1` (sin el sufijo `.0`) ya falla en vez de ignorarse.
- **Off-by-one 106/16/89** (§3.3) — la tarjeta declaraba 106 features tras codificación cuando 106 son *columnas* e incluyen TARGET. Ahora dice 105 y la resta cierra: 105 − 16 = 89.
- **RZ → P** (§5.3) — el texto decía "puerta RZ" y el pie del diagrama "H + P". Unificado en P, que es lo que usa el `ZZFeatureMap` y lo que muestra la figura.
- **Importancias RF duplicadas** (§1.11) — `QSVM_FEATURES` llevaba 0,2452 / 0,1853 / 0,0325… y `RF_TOP8_IMPORTANCE` 0,2454 / 0,1855 / 0,0323…. Los del notebook 03 son los segundos. Ahora `QSVM_FEATURES` lleva los valores del notebook y **`RF_TOP8_IMPORTANCE` se deriva de él** — una sola fuente, no pueden volver a divergir.
- **Métricas reconciliadas contra los `.npy`** (§2.4) — ver abajo.

~~Decidir entre 21,1 y 22,6 min de entrenamiento~~ — resuelto: 21,1 es el correcto y la app ya lo mostraba (§5.2).

---

## 9. Reconciliación automática de métricas

El cambio de fondo que pedía la consistencia: las métricas dejan de estar **solo** transcritas. `reconciliar_metricas()` recalcula AUC (Mann-Whitney, sin dependencias externas), matriz de confusión, accuracy, F1-macro y MCC de los tres modelos desde sus scores por instancia, cada uno **en su propio umbral**, y los compara con lo declarado en `MODELS`.

El resultado se pinta en la página de Resultados:

- **✓ Reconciliadas** — las doce cifras coinciden (estado actual, verificado en pantalla).
- **⚠ Sin reconciliar** — se listan las discrepancias exactas, modelo a modelo.

Si alguien reentrena y sustituye un `.npy` sin tocar las constantes, **la app lo dice sola** en vez de seguir mostrando números muertos. Es el mecanismo que habría detectado por sí solo la desincronización que este informe buscaba a mano.

Detalle del umbral del SVM: su corte real es el signo de `decision_function`, que no se persistió. Su equivalente en la escala de probabilidad guardada cae entre **0,221729** (último score negativo) y **0,225167** (primero positivo); se almacena el punto medio (0,223448) para que la reconciliación no dependa de redondeos. Con el valor redondeado a 0,2217 fallaba por un solo caso (tn=1249 en vez de 1250).

---

## 9-bis. Fidelidad del camino de inferencia frente a los notebooks

Pregunta del autor (6 ago 2026): *¿los valores que muestra la app son los que producen realmente los modelos entrenados?* Es la pregunta correcta: la coherencia interna no sirve de nada si la app alimenta los modelos de forma distinta a como se entrenaron. Cadena de custodia verificada eslabón a eslabón:

| Eslabón | Evidencia | Estado |
|-|-|-|
| **Política de escalado** | NB03 celda 15: *«StandardScaler · Ajuste solo sobre train · **Solo para SVM y QSVM**»*. NB04 carga `X_train_lgbm.parquet` (crudo); NB05 carga `X_train_svm.parquet`, que la celda 26 escribió desde `X_train_svm_scaled` | ✓ la app escala **solo** el SVM |
| **Procedencia del scaler** | NB03 celda 17 escribe `scaler_correcto.json` directamente de `scaler.mean_` / `scaler.scale_`, ajustados solo sobre train | ✓ no es una reconstrucción |
| **Valores exactos** | El notebook imprimió `Media HbA1c 5.7185 · Std 0.7707 · Mediana 5.50`; los ficheros de la app llevan 5,7185 / 0,7707 / 5,50 | ✓ idénticos |
| **Medianas** | `X_train_svm[c].median()` sobre datos **crudos**, que es lo que necesita el formulario | ✓ correcto |
| **PKL → ONNX** | NB04 y NB05 imprimen ambos *«Coincidencia PKL vs ONNX: 100.0 % · ONNX verificado OK»* | ✓ verificado en origen |
| **Scores `.npy`** | Exportados de `y_pred_proba` con AUC de control 0,9485; la reconciliación de la app recalcula 0,9485 desde ellos | ✓ bucle cerrado |
| **Lectura del tensor** | LightGBM (`zipmap:False`) devuelve `ndarray (N,2)`; SVM devuelve `seq(map(int64,float))`. Ambas sumas de probabilidad dan 1,000000 | ✓ `out[1][i][1]` acierta en ambos |

> **Trampa de nomenclatura documentada.** `X_train_svm.parquet` y `X_test_svm.parquet` **contienen datos escalados** pese a que el nombre no lo diga (NB03 celda 26 los escribe desde `X_train_svm_scaled` / `X_test_svm_scaled`). Cualquiera que reutilice esos ficheros asumiendo que son crudos los escalaría dos veces.

**Anomalía encontrada al verificar: el opset del LightGBM está mal declarado.** Parseando los ficheros:

| Fichero | `ai.onnx` | `ai.onnx.ml` |
|-|-|-|
| `svm_final.onnx` | 9 | 1 |
| `lgbm_final.onnx` | **1** ⚠ | 3 |

El NB04 pidió `target_opset={'': 12, 'ai.onnx.ml': 3}`, pero el fichero declara 1 para el dominio por defecto — rareza conocida del conversor de LightGBM de `onnxmltools`. Afecta **solo** al modelo que mueve el Predictor. Hoy da resultados correctos (verificado 100 % contra el pickle), pero es lo que el pendiente C4 debe cerrar.

**Límite de la verificación, y cómo se cierra.** No es posible comprobar localmente el trayecto completo —pasar `X_test` por la app y reproducir `lgbm_y_scores.npy` valor a valor— porque **el conjunto de test no está en el repositorio**. Se añade el mecanismo para cerrarlo:

- `notebooks/INSTRUCCIONES_exportar_golden_set.md` — dos celdas nuevas (NB04 y NB05) que exportan 25 filas del test con la probabilidad del modelo entrenado. **No son simétricas**: el NB04 ya tiene datos crudos; el NB05 debe invertir el escalado con el propio scaler, porque el golden set tiene que llevar valores crudos para que sea la app quien escale — que es justo el paso a verificar.
- `verificar_golden()` en `app.py` — reproduce el camino completo y compara con tolerancia 1 × 10⁻⁴ (ruido esperado en `float32`: ~10⁻⁷). El resultado se publica en Gobernanza → Linaje.

Los tres estados están probados con golden sets sintéticos temporales, borrados después:

| Escenario | Resultado |
|-|-|
| Ficheros ausentes (estado actual del repo) | *«Sin verificar»* + instrucciones. **Nunca afirma haber comprobado lo que no ha comprobado** |
| Coincidencia | *«✓ Camino de inferencia verificado — 25 filas · dif. máx. 0,00e+00»* |
| Leer la clase equivocada del tensor (`p → 1−p`) | *«⚠ DISCREPA · dif. máx. 9,85e-01»* |
| Olvidar el escalado del SVM (deriva de 0,004) | *«⚠ DISCREPA · dif. máx. 4,00e-03»* |

---

## 10. Verificación posterior a los cambios

- `ast.parse` sobre las 2.998 líneas: correcto.
- **Las 7 páginas renderizadas sin excepción** con `streamlit.testing.v1.AppTest`, comprobando además que cada una emite su contenido propio (no siete veces la misma).
- Predictor en Vivo: 8 sliders presentes, vector de 89 features sin NaN, cinco grupos one-hot sumando 1.
- Página de Resultados: sello **"✓ Reconciliadas"** en pantalla y los tres umbrales visibles (`p ≥ 0,50`, `p ≈ 0,22`, `df > 0`).
- Servidor recargado sin trazas de error (solo los avisos preexistentes de opset 1 — ver C4).

---

## Anexo — Reproducibilidad de esta auditoría

Tres scripts en [`audit_scripts/`](audit_scripts/), ejecutados con el mismo `onnxruntime` que usa la app. No toman argumentos y resuelven `streamlit/models/` de forma relativa a su propia ubicación, así que funcionan en cualquier clon del repositorio:

```bash
python audit_scripts/audit_sens.py
python audit_scripts/audit_metrics.py
python audit_scripts/audit_baseline.py
```

- `audit_sens.py` — barridos de sensibilidad de las 8 variables (grueso 0,1 y fino 0,01) replicando `predict_real()` línea a línea.
- `audit_metrics.py` — recálculo de AUC (trapecio y Mann-Whitney), matrices de confusión, accuracy, F1-macro y MCC sobre los `.npy`, más búsqueda del umbral que reproduce cada CM declarada.
- `audit_baseline.py` — auditoría del vector base (one-hot, moda vs mediana), sensibilidad a WTINT2YR, perfiles clínicos y comparación slider vs distribución de entrenamiento.

Advertencia de `onnxruntime` en los tres: los `.onnx` están sellados con **opset 1**, por debajo del 7 que la librería garantiza. Hoy funcionan por soporte legacy; una actualización de `onnxruntime` puede romperlos. Conviene reexportar con un opset moderno antes de archivar el TFM.
