# Notas técnicas — tfm-qml-databricks-pipeline

Este documento recoge las limitaciones técnicas encontradas durante la implementación, las decisiones de diseño adoptadas como respuesta y los workarounds aplicados. Está dirigido al tutor, tribunal evaluador y cualquier investigador que desee reproducir el pipeline.

---

## 1. Entorno de ejecución

**Databricks Community Edition (Serverless)**

El entorno serverless de Databricks Community Edition impone restricciones que no están presentes en ediciones de pago ni en entornos locales. Estas restricciones condicionaron decisiones de arquitectura relevantes documentadas a continuación.

| Parámetro | Valor |
|-----------|-------|
| Plataforma | Databricks Community Edition |
| Modalidad | Serverless (sin cluster dedicado) |
| Python | 3.12 |
| pandas | 1.5.3 (fijada por el entorno) |
| numpy | 1.23.5 (fijada por el entorno) |
| RAM disponible | ~15 GB (variable según disponibilidad) |

---

## 2. Limitaciones técnicas y soluciones implementadas

### 2.1 `spark.conf` bloqueado en Serverless

**Problema:** La configuración de credenciales AWS mediante `spark.conf.set("fs.s3a.access.key")` está bloqueada en Databricks serverless con el error `CONFIG_NOT_AVAILABLE (SQLSTATE: 42K0I)`. Este mecanismo es el estándar para conectar Spark directamente con S3.

**Impacto:** Las capas Bronze, Silver y Gold no pueden residir directamente en S3. Deben almacenarse en Unity Catalog Volumes (`/Volumes/workspace/default/nhanes/`).

**Solución implementada:** Se utilizó `boto3` como cliente alternativo para leer los archivos XPT desde S3 y escribirlos en Unity Catalog Volumes. El pipeline mantiene S3 como capa de almacenamiento de origen (raw) y Unity Catalog como capa de procesamiento.

```python
# Alternativa funcional a spark.conf en serverless
s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="eu-west-1"
)
```

---

### 2.2 MLflow bloqueado en Serverless

**Problema:** La integración nativa de MLflow con Databricks está deshabilitada en la capa gratuita. Cualquier llamada a `mlflow.start_run()` o `mlflow.log_metric()` produce errores de autenticación.

**Impacto:** No es posible registrar experimentos, métricas ni artefactos en el servidor MLflow nativo de Databricks.

**Solución implementada:** Trazabilidad mediante dos mecanismos alternativos:
- **Delta Lake transaction logs:** cada escritura genera automáticamente un log ACID con versión, timestamp y métricas de operación.
- **CSV de métricas:** cada notebook de modelado persiste sus métricas en un archivo CSV en Unity Catalog Volumes (`metricas_lgbm.csv`, `metricas_svm.csv`, `metricas_qsvm.csv`).

---

### 2.3 Great Expectations incompatible con el entorno

**Problema:** Great Expectations requiere pandas >= 1.4 con compatibilidad completa con el módulo `dataset`, incompatible con la versión fijada pandas 1.5.3 / numpy 1.23.5 del entorno serverless.

**Impacto:** No es posible usar Great Expectations como framework de validación de calidad del dato.

**Solución implementada:** Se utilizó `dataframe-expectations==0.7.0` como alternativa compatible. Se implementó una suite de 15 expectativas sobre la capa Silver cubriendo tres dimensiones: completitud (5 expectativas), rangos clínicos válidos (8 expectativas) y volumen esperado (2 expectativas). Resultado: 15/15 passed, pass rate 1.0.

---

### 2.4 QSVM — Coste computacional O(n²)

**Problema:** El cálculo de la matriz de kernel cuántico tiene complejidad O(n²) respecto al número de instancias de entrenamiento. Con los 6.264 registros del conjunto de entrenamiento completo, el número de evaluaciones del circuito ascendería a ~39 millones de operaciones, inviable en un simulador clásico dentro del plazo académico.

**Impacto:** No es posible entrenar el QSVM sobre el conjunto de entrenamiento completo.

**Solución implementada:** Entrenamiento sobre muestra estratificada de 500 instancias (430 negativos / 70 positivos, ratio 86%/14% preservado). La evaluación se realiza sobre el conjunto de test completo de 1.567 instancias para garantizar comparabilidad de métricas con los modelos clásicos.

**Tabla de escalabilidad empírica:**

| Instancias train | Operaciones kernel | Tiempo entrenamiento | Support Vectors |
|-----------------|-------------------|---------------------|-----------------|
| 100 | 10.000 | ~1 min | [86, 14] |
| 300 | 90.000 | ~6 min | [258, 42] |
| 500 | 250.000 | ~22 min | [426, 70] |
| 800 | 640.000 | ~60 min (estimado) | — |
| 1.500 | 2.250.000 | OOM (kernel crash) | — |

**Predicción por lotes:** La evaluación sobre el conjunto de test también requiere procesamiento por lotes de 100 instancias para evitar saturación de memoria. Tiempo total de predicción: ~132 minutos.

---

### 2.5 QSVM — Sin soporte ONNX nativo

**Problema:** El formato ONNX no soporta operaciones cuánticas. Los conversores disponibles (`skl2onnx`, `onnxmltools`) no pueden serializar el kernel cuántico basado en simulación de estados cuánticos.

**Impacto:** El QSVM no puede exportarse en formato portable ONNX como los modelos clásicos.

**Solución implementada:** Serialización mediante `joblib.dump()` (formato pickle) con un wrapper de compatibilidad. El modelo serializado incluye la referencia al kernel cuántico pero requiere el entorno Qiskit para inferencia.

---

### 2.6 Versiones de Qiskit — Incompatibilidad con versiones fijadas

**Problema:** El archivo de restricciones de paquetes de Databricks serverless (`immutable_package_constraints.txt`) bloquea la instalación de versiones específicas de Qiskit (`qiskit==0.45.0`).

**Impacto:** No es posible garantizar reproducibilidad exacta con una versión fijada de Qiskit.

**Solución implementada:** El pipeline corre con las versiones disponibles en el entorno serverless:

| Librería | Versión utilizada |
|----------|-------------------|
| qiskit | 2.5.0 |
| qiskit-machine-learning | 0.9.0 |
| qiskit-algorithms | 0.4.0 |

La API de `ZZFeatureMap`, `FidelityQuantumKernel` y `ComputeUncompute` es compatible entre estas versiones.

---

### 2.7 Pérdida de variables por duración de sesión

**Problema:** Las celdas de larga duración (entrenamiento ~22 min + predicción ~132 min) pueden causar pérdida de variables en memoria al finalizar, especialmente en sesiones serverless con timeout.

**Impacto:** Variables como `y_scores`, `qsvm_model` y `y_pred` pueden no estar disponibles para celdas posteriores.

**Solución implementada:** Persistencia inmediata tras cada operación costosa:

```python
# Tras entrenamiento
joblib.dump(qsvm_model, f"{models_dir}/qsvm_final.pkl")

# Tras predicción
np.save(f"{models_dir}/qsvm_y_scores.npy", y_scores)
np.save(f"{models_dir}/qsvm_y_test.npy",   y_test.values)
```

El notebook implementa `TRAINING_MODE = False` para recargar desde disco en ejecuciones posteriores sin repetir el entrenamiento.

---

### 2.8 Winsorización IQR aplicada a variables categóricas codificadas

**Problema:** La winsorización IQR×3 de la capa Silver (`notebook_02_silver.ipynb`) se aplica a todas las variables numéricas salvo cinco (`SEQN`, `TARGET`, `DIQ010`, `RIDAGEYR`, `RIAGENDR`). NHANES codifica numéricamente un gran número de variables categóricas y ordinales —respuestas sí/no como 1/2, códigos 7 ("no sabe") y 9 ("rehúsa responder"), idioma de la entrevista, etc.—, por lo que estas quedan incluidas en el tratamiento pese a no ser variables continuas.

**Mecanismo:** En una variable donde más del 75% de las observaciones comparte un mismo valor se cumple Q1 = Q3 y, por tanto, IQR = 0. Los límites de winsorización se colapsan sobre un único punto (`lower = Q1 − 3·0 = Q1`, `upper = Q3 + 3·0 = Q1`) y `Series.clip()` lleva **todas** las observaciones a ese valor: la variable se convierte en constante.

**Impacto medido:** 24 de las 91 columnas de la capa Silver (26,4%) presentan varianza cero. De ellas, 10 corresponden a variables categóricas colapsadas por este mecanismo:

`PAQ605`, `PAQ635`, `PAQ650`, `DMDCITZN`, `DMQMILIZ`, `SIALANG`, `FIALANG`, `MIALANG`, `AIALANGA`, `DMDHHSZA`

Las 14 restantes (`RIDSTATR`, `BMDSTATS`, `BPXPULS`, `BPAEN1`–`BPAEN3`, entre otras) son constantes al menos en parte como consecuencia natural de los filtros de la capa Silver —población adulta examinada en ayunas—, sin que sea posible separar ambas causas sin recuperar los datos previos a la winsorización.

La correspondencia es verificable en la propia salida del notebook: `PAQ635` reporta 1.929 outliers corregidos, que son exactamente las 1.929 observaciones cuyo valor difería del valor modal y que quedaron llevadas a él.

**Impacto sobre el modelado:** Las columnas de varianza cero se propagan a la capa Gold y forman parte del conjunto de 89 features. **No introducen sesgo ni fuga de información**: al ser constantes, su contribución a la predicción es nula. `StandardScaler` las transforma en columnas de ceros (scikit-learn asigna `scale_ = 1` cuando la desviación típica es cero) y ninguno de los tres modelos puede extraer señal de ellas. El efecto es, por tanto, una **pérdida de información potencialmente útil**, no una distorsión de los resultados obtenidos.

**Decisión adoptada:** Se documenta la limitación sin modificar el pipeline. Corregirla alteraría la capa Silver y, en cascada, la capa Gold y los tres modelos, invalidando los resultados experimentales ya obtenidos. El tratamiento correcto —recogido como línea de trabajo futura— consistiría en excluir las variables categóricas del proceso y condicionar la winsorización a la existencia de dispersión intercuartílica:

```python
# Winsorización restringida a variables continuas con dispersión no nula
vars_winsorizar = [v for v in vars_numericas
                   if v not in vars_excluir_winsor
                   and v not in vars_categoricas_presentes]

for col in vars_winsorizar:
    Q1, Q3 = df_silver[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    if IQR == 0:          # sin dispersión intercuartílica: no winsorizar
        continue
    ...
```

Su aplicación exigiría re-entrenar y re-evaluar los tres modelos sobre el conjunto de features resultante.

---

### 2.9 Filtro de correlación calculado antes de la partición train/test

**Problema:** La eliminación de variables con correlación r > 0.90 (`notebook_03_gold.ipynb`, CELDA 5) se calcula sobre el dataset completo, antes de la partición 80/20. Las 16 columnas descartadas se determinan, por tanto, usando también las observaciones que después forman el conjunto de test.

**Impacto:** Constituye una forma leve de fuga de información por selección de características. No afecta al escalado —el `StandardScaler` se ajusta exclusivamente sobre train (CELDA 7)— ni a la selección de las 8 variables del QSVM, que emplea un `RandomForestClassifier` entrenado solo con `X_train_svm_scaled` (CELDA 8). El sesgo esperado es pequeño, porque la correlación entre pares de predictores es una propiedad estructural del dataset y no depende de la variable objetivo, pero la selección deja de ser estrictamente ciega al conjunto de test.

**Decisión adoptada:** Se documenta sin modificar. Recalcular la matriz de correlación solo sobre train cambiaría el conjunto final de features y, en cascada, los tres modelos. El tratamiento correcto sería particionar primero y derivar `cols_alta_corr` únicamente de `X_train`.

---

### 2.10 Duplicación del peso de muestreo en el merge intracíclico

**Problema:** Los módulos GLU, INS y TRIGLY incluyen la variable `WTSAF2YR` (peso de muestreo del subgrupo en ayunas). El join intracíclico del NB01 genera en consecuencia tres columnas: `WTSAF2YR`, `WTSAF2YR_x` y `WTSAF2YR_y`. La lista de exclusión del NB03 (CELDA 3) solo casa con el nombre exacto, por lo que las variantes con sufijo entran inicialmente en el conjunto de features.

**Impacto:** Las tres variantes de `WTSAF2YR` acaban descartadas antes del modelado — la exacta por exclusión explícita, y las `_x`/`_y` por el filtro de correlación r > 0.90 de la CELDA 5, junto con `WTMEC2YR`. El mecanismo, sin embargo, es accidental: si la correlación entre ambas variantes hubiera quedado por debajo del umbral, habrían entrado como predictores.

**`WTINT2YR` sí llega al modelado.** El peso de muestreo de la entrevista no figura en la lista de exclusión y su correlación con `WTMEC2YR` supera el umbral, pero el filtro descarta la *segunda* columna de cada par correlacionado: `WTMEC2YR` se elimina y `WTINT2YR` sobrevive. Es, por tanto, **una de las 89 features** con las que se entrenan los tres modelos. Su correlación con la variable objetivo es −0,1118 y su rango, 4.363–137.870.

Un peso de muestreo no es una variable clínica: codifica la probabilidad de selección de cada individuo en el diseño muestral de NHANES, no una característica del paciente. Su presencia entre los predictores es un artefacto del pipeline, no una decisión metodológica. No introduce fuga de la variable objetivo, pero sí permite al modelo apoyarse en información del diseño de la encuesta.

**Decisión adoptada:** No se modifica la lista de exclusión. Añadir `WTINT2YR` cambiaría el conjunto de features y, en cascada, los tres modelos y todas las métricas publicadas. Se incorpora en su lugar una comprobación en la CELDA 6 que documenta la situación conocida y detecta la aparición de cualquier *otro* peso de muestreo:

```python
pesos = [c for c in X_svm.columns if c.startswith(("WTSAF", "WTMEC", "WTINT"))]
assert set(pesos) <= {"WTINT2YR"}, f"Peso de muestreo inesperado: {pesos}"
```

Excluir `WTINT2YR` y re-evaluar los tres modelos queda identificado como línea de trabajo futura, junto con la corrección de la winsorización (2.8).

---

### 2.11 El modelo QSVM serializado no es recargable entre versiones de Qiskit

**Problema:** `qsvm_final.pkl` se genera con `joblib.dump()`. El objeto `SVC` almacena como función kernel `quantum_kernel.evaluate`, lo que arrastra al pickle el `ZZFeatureMap` completo con sus objetos `ParameterExpression`. Al recargarlo en un entorno cuya versión de Qiskit difiere de la que lo generó, la deserialización falla:

```
TypeError: ParameterExpression.__new__() missing 2 required positional
arguments: 'name_map' and 'expr'
```

**Impacto:** El atajo `TRAINING_MODE = False` —cargar el modelo en segundos en lugar de re-entrenar durante horas— **no funciona** cuando el entorno serverless ha actualizado Qiskit respecto a la sesión de entrenamiento. Dado que Databricks Serverless actualiza sus paquetes sin previo aviso (ver 2.6), es una situación previsible y no excepcional.

**Solución implementada:** La CELDA 6 envuelve la carga en un `try/except`. Si la deserialización falla, se informa del motivo y `TRAINING_MODE` pasa a `True`, de modo que el notebook re-entrena en lugar de abortar. Combinado con la autodetección del fichero, el notebook queda operativo en los tres escenarios posibles:

| Estado del volumen | `TRAINING_MODE` | Resultado |
|-|-|-|
| Sin `qsvm_final.pkl` | `True` | Entrena desde cero (~22 min) |
| Con `.pkl` legible | `False` | Carga el modelo (segundos) |
| Con `.pkl` de otra versión de Qiskit | `False` → `True` | Detecta el fallo y re-entrena |

**Nota sobre los resultados almacenados:** las celdas condicionales del NB06 conservan salidas de dos sesiones distintas. Tanto las ramas `if TRAINING_MODE:` (entrenamiento y evaluación por lotes) como la rama `if not TRAINING_MODE:` (carga de scores desde disco) tienen resultados, pese a ser mutuamente excluyentes. Las métricas son coherentes entre sí —los scores cargados proceden del entrenamiento registrado—, pero una reejecución completa en un único modo unificaría las salidas.

---

## 3. Componentes cuánticos — explicación detallada

### 3.1 ZZFeatureMap

El `ZZFeatureMap` es el mapa de características cuántico que transforma cada instancia de datos en un estado cuántico. Recibe un vector de `n` valores numéricos (en este TFM, 8 variables clínicas) y los codifica como ángulos de rotación en un circuito cuántico de `n` qubits.

El circuito aplica tres tipos de puertas en secuencia:

- **Puertas Hadamard (H):** crean superposición inicial en cada qubit, poniendo el sistema en una combinación de `|0⟩` y `|1⟩` simultáneamente.
- **Puertas de fase P(2x[i]):** codifican cada variable clínica `x[i]` como un ángulo de rotación en su qubit correspondiente.
- **Puertas de entrelazamiento P(2(π-x[i])(π-x[j])):** generan correlaciones cuánticas entre pares de qubits adyacentes, capturando interacciones entre pares de variables clínicas que un kernel clásico no puede representar directamente.

El circuito se repite `reps=2` veces para aumentar la expresividad del mapa de características. Con `entanglement=linear`, cada qubit se entrelaza solo con su vecino inmediato, manteniendo el coste de simulación dentro de límites razonables para 8 qubits.

### 3.2 FidelityQuantumKernel

El `FidelityQuantumKernel` calcula la función kernel entre dos puntos `x` e `y` como la fidelidad cuántica entre sus estados cuánticos correspondientes:

```
K(x, y) = |⟨ψ(x)|ψ(y)⟩|²
```

Donde `|ψ(x)⟩` es el estado cuántico generado al aplicar el `ZZFeatureMap` al punto `x`. Esta medida de similitud opera en el espacio de Hilbert generado por el circuito cuántico, cuya dimensionalidad crece exponencialmente con el número de qubits (2⁸ = 256 dimensiones para 8 qubits).

### 3.3 ComputeUncompute

`ComputeUncompute` es el algoritmo que implementa el cálculo de la fidelidad cuántica. Su nombre describe exactamente lo que hace:

**Paso 1 — Compute:** aplica el circuito `ZZFeatureMap` con los datos del punto `x`, generando el estado cuántico `|ψ(x)⟩`.

**Paso 2 — Uncompute:** aplica el circuito inverso (adjunto) con los datos del punto `y`, que equivale a "deshacer" el estado `|ψ(y)⟩` hacia el estado base `|0⟩`.

**Medición:** la probabilidad de medir `|0...0⟩` al final del circuito combinado es exactamente `|⟨ψ(x)|ψ(y)⟩|²`, que es el valor del kernel entre `x` e `y`.

En el código del TFM los tres componentes se encadenan así:

```python
# Simulador cuántico exacto sin ruido
sampler = Sampler()

# Algoritmo de fidelidad basado en Compute-Uncompute
fidelity = ComputeUncompute(sampler=sampler)

# Kernel cuántico que usa la fidelidad como función de similitud
quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map,   # ZZFeatureMap (8 qubits, reps=2, linear)
    fidelity=fidelity           # ComputeUncompute
)

# El SVC usa quantum_kernel.evaluate como función kernel
qsvm_model = SVC(
    kernel=quantum_kernel.evaluate,
    C=1.0,
    class_weight="balanced",
    probability=True
)
```

La elección de `StatevectorSampler` (simulador de vector de estado exacto) garantiza resultados deterministas y reproducibles, ya que simula el estado cuántico completo sin ruido de medición. Esto es coherente con el objetivo del TFM de evaluar la capacidad discriminativa del kernel cuántico en condiciones ideales, siguiendo el protocolo de Havlíček et al. (2019).

---

## 4. Decisiones de diseño relevantes

### 4.1 Partición train/test

Se eligió partición 80/20 estratificada por clase objetivo por dos razones: es el estándar en los estudios de referencia sobre predicción clínica con NHANES, y maximiza el conjunto de entrenamiento en un dataset de tamaño moderado. La estratificación garantiza que la proporción de casos positivos es idéntica en train (14.03%) y test (14.04%).

### 4.2 Gestión del desbalance de clases

Se optó por `class_weight='balanced'` en LightGBM y SVM-RBF, descartando SMOTE para preservar la auditabilidad del pipeline y la distribución real de la población. El desbalance 86%/14% refleja la prevalencia real de diabetes tipo 2 en la población NHANES.

### 4.3 Selección de features para QSVM

Las 8 features del QSVM se seleccionaron mediante `RandomForestClassifier` sobre el conjunto Gold, excluyendo explícitamente las variables DIQ de tratamiento para evitar leakage conceptual. El límite de 8 qubits responde al coste de simulación: circuitos de más de 8-10 qubits son simulables pero con tiempos prohibitivos en CPU.

### 4.4 Evaluación del QSVM mediante `decision_function`

Se usó `decision_function` en lugar de `predict_proba` para calcular el AUC-ROC del QSVM porque produce scores continuos más informativos para la curva ROC, especialmente en modelos con bajo poder discriminativo como el QSVM en este contexto.

---

## 5. Reproducibilidad

Para reproducir el pipeline completo:

1. Descargar los 27 archivos XPT de NHANES desde https://wwwn.cdc.gov/nchs/nhanes/
2. Subirlos al bucket S3 en la carpeta `raw/`
3. Dar de alta las credenciales AWS en un scope de Databricks Secrets (`aws-nhanes`); `notebook_01_bronze.ipynb` las recupera con `dbutils.secrets.get()`
4. Ejecutar los notebooks en orden (ver README.md)
5. Para el QSVM, usar `TRAINING_MODE = True` en la primera ejecución

**Semilla de aleatoriedad:** `random_state=42` en todos los modelos y splits.

**Modo de ejecución del QSVM:** `notebook_06_qsvm.ipynb` detecta automáticamente si existe `qsvm_final.pkl` en el volumen. Si no existe, entrena desde cero; si existe, lo carga. El valor puede forzarse manualmente para re-entrenar sobre un modelo ya guardado.

**Valores SHAP del SVM-RBF:** a diferencia de LightGBM, el SVM-RBF no admite `TreeExplainer` y requiere `KernelExplainer`, cuyo coste es de varias horas. La CELDA 9 del NB05 calcula los valores sobre una muestra de 200 instancias de test y los persiste en `shap_values_svm.npy` y `shap_X_test_sample_svm.npy`; en ejecuciones posteriores los reutiliza desde disco.

---

## 6. Despliegue de la aplicación Streamlit

La aplicación no lee de Unity Catalog Volumes: Streamlit Community Cloud solo accede al contenido del repositorio. Tras ejecutar los notebooks de modelado, los artefactos deben copiarse manualmente desde `/Volumes/workspace/default/nhanes/models/` a `streamlit/models/`.

| Artefacto | Lo genera | Uso en la aplicación |
|-|-|-|
| `lgbm_final.onnx` | NB04 | Inferencia del Live Predictor |
| `svm_final.onnx` | NB05 | Inferencia del Live Predictor |
| `scaler_correcto.json` | NB03 | Escalado de las entradas del formulario |
| `medianas_correctas.json` | NB03 | Valores por defecto de las 81 variables no editables |
| `lgbm_y_scores.npy` | NB04 | Curva ROC |
| `svm_y_scores.npy` | NB05 | Curva ROC |
| `qsvm_y_scores.npy` | NB06 | Curva ROC |
| `qsvm_y_test.npy` | NB06 | Etiquetas del test, compartidas por los tres modelos |

**Etiquetas compartidas:** el NB04 y el NB05 generan también `lgbm_y_test.npy` y `svm_y_test.npy`, pero no se despliegan. Los tres modelos se evalúan sobre la misma partición de test, por lo que `qsvm_y_test.npy` sirve para todos. La función `_load_roc_scores()` de `app.py` comprueba que la longitud de scores y etiquetas coincide y devuelve `None` en caso contrario: una partición futura de distinto tamaño impediría dibujar la curva en lugar de dibujarla mal.

**Nomenclatura obsoleta:** en el volumen pueden coexistir `scaler_params.json` y `medianas_train.json`, procedentes de una versión anterior del pipeline. Los ficheros vigentes son `scaler_correcto.json` y `medianas_correctas.json`, generados por la CELDA 7 del NB03.

---

*Documento técnico complementario al TFM — Universidad Europea de Valencia — 2025-2026*
