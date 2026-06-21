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
| qiskit | 2.4.1 |
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

### 3.1 Partición train/test

Se eligió partición 80/20 estratificada por clase objetivo por dos razones: es el estándar en los estudios de referencia sobre predicción clínica con NHANES, y maximiza el conjunto de entrenamiento en un dataset de tamaño moderado. La estratificación garantiza que la proporción de casos positivos es idéntica en train (14.03%) y test (14.04%).

### 3.2 Gestión del desbalance de clases

Se optó por `class_weight='balanced'` en LightGBM y SVM-RBF, descartando SMOTE para preservar la auditabilidad del pipeline y la distribución real de la población. El desbalance 86%/14% refleja la prevalencia real de diabetes tipo 2 en la población NHANES.

### 3.3 Selección de features para QSVM

Las 8 features del QSVM se seleccionaron mediante `RandomForestClassifier` sobre el conjunto Gold, excluyendo explícitamente las variables DIQ de tratamiento para evitar leakage conceptual. El límite de 8 qubits responde al coste de simulación: circuitos de más de 8-10 qubits son simulables pero con tiempos prohibitivos en CPU.

### 3.4 Evaluación del QSVM mediante `decision_function`

Se usó `decision_function` en lugar de `predict_proba` para calcular el AUC-ROC del QSVM porque produce scores continuos más informativos para la curva ROC, especialmente en modelos con bajo poder discriminativo como el QSVM en este contexto.

---

## 4. Reproducibilidad

Para reproducir el pipeline completo:

1. Descargar los 27 archivos XPT de NHANES desde https://wwwn.cdc.gov/nchs/nhanes/
2. Subirlos al bucket S3 en la carpeta `raw/`
3. Configurar credenciales AWS en `notebook_01_bronze.ipynb`
4. Ejecutar los notebooks en orden (ver README.md)
5. Para el QSVM, usar `TRAINING_MODE = True` en la primera ejecución

**Semilla de aleatoriedad:** `random_state=42` en todos los modelos y splits.

---

*Documento técnico complementario al TFM — Universidad Europea de Valencia — 2025-2026*
