# Informe de auditoría — Pipeline TFM QML NHANES

**Proyecto:** `tfm-qml-databricks-pipeline`
**Autor del TFM:** Juan Albornoz Carrasco — Universidad Europea de Valencia
**Fechas de la auditoría:** 2 – 3 de agosto de 2026
**Alcance:** 7 notebooks del pipeline, `streamlit/app.py`, `README.md`, `TECHNICAL_NOTES.md`, `requirements.txt`

> **Auditoría complementaria.** Este informe es análisis **estático** (ningún notebook se ejecutó). En [INFORME\_AUDITORIA\_DASHBOARD.md](INFORME_AUDITORIA_DASHBOARD.md) hay una auditoría **dinámica** posterior del dashboard: se ejecutaron los modelos ONNX y se recalcularon todas las métricas contra los `.npy`. Varias de las limitaciones que aquí se documentan sin verificar quedan allí confirmadas con cifras — en particular el daño real de la winsorización sobre variables categóricas (sección 2) y el peso de `WTINT2YR` entre las features.

---

## ⚠️ Limitación fundamental de esta auditoría

**Ningún notebook ha sido ejecutado.** La auditoría se realizó exclusivamente mediante análisis estático desde un entorno local Windows, sin acceso a Databricks, al bucket S3 ni al cluster.

Lo que **sí** está verificado mecánicamente:

- Validez del formato `nbformat` de los 7 notebooks.
- Compilación sintáctica de las 113 celdas de código (`ast.parse`).
- Preservación de los outputs originales, comparados byte a byte contra copia de seguridad.
- Ausencia de credenciales y datos personales.
- Coherencia de nombres de artefactos entre notebooks.
- Los 10 `assert` introducidos, simulados uno a uno contra los datos reales del proyecto.

Lo que **no** está verificado:

- Que cualquier celda se ejecute correctamente en Databricks.
- El comportamiento en tiempo de ejecución de las líneas de código nuevas.

**Conclusión: es imprescindible ejecutar el pipeline completo antes de publicar la versión final.** El plan de prueba está en la sección 8.

---

## 1. Resumen ejecutivo

Se auditaron los 7 notebooks del pipeline Medallón (Bronze → Silver → Gold → 3 modelos → validación) y la aplicación Streamlit.

| Categoría | Detectados | Corregidos | Documentados sin corregir |
|-|-|-|-|
| Roturas de reproducibilidad | 8 | 8 | — |
| Seguridad y privacidad | 3 | 3 | — |
| Hallazgos metodológicos | 4 | — | 4 |
| Calidad de código | ~40 | ~40 | — |
| Errores en documentación | 5 | 5 | — |

**Estado del pipeline antes de la auditoría:** no reproducible desde cero. Cinco eslabones rotos impedían una ejecución limpia en orden.

**Estado después:** cadena completa y coherente, con todos los outputs originales intactos salvo tres eliminaciones documentadas.

**Los resultados experimentales no se han alterado.** Ninguna corrección modifica datos, particiones, hiperparámetros ni métricas. Los hallazgos que habrían exigido cambiarlos se documentaron en `TECHNICAL_NOTES.md` en lugar de aplicarse.

---

## 2. Estado final de los ficheros

| Fichero | Celdas | Código | Outputs | Tamaño | MD5 |
|-|-|-|-|-|-|
| `notebook_01_bronze.ipynb` | 27 | 15 | 14 | 69,4 KB | `6DFACE0C` |
| `notebook_02_silver.ipynb` | 38 | 19 | 16 | 39,2 KB | `4FDBD21B` |
| `notebook_03_gold.ipynb` | 35 | 19 | 17 | 48,6 KB | `060B1F1E` |
| `notebook_04_lgbm.ipynb` | 35 | 18 | 15 | 349,7 KB | `413783A7` |
| `notebook_05_svm.ipynb` | 28 | 15 | 14 | 331,1 KB | `298582C3` |
| `notebook_06_qsvm.ipynb` | 38 | 21 | 26 | 422,7 KB | `2676B041` |
| `notebook_07_validacion_silver.ipynb` | 13 | 6 | 4 | 13,0 KB | `6612E44A` |

Documentación: `README.md` (8,6 KB), `TECHNICAL_NOTES.md` (23,0 KB), `requirements.txt` (1,0 KB).

Las copias de `NHANES\Notebooks de Databricks\` están sincronizadas con las del repositorio (mismos hashes).

**Diagnósticos del IDE: 0** en los 7 notebooks.

---

## 3. Roturas de reproducibilidad corregidas

### 3.1 `bronze_path` sin definir — NB01

La variable se usaba en 4 puntos (celdas de escritura Delta, historial y validación) sin definirse en ninguna parte. Funcionaba solo porque quedaba viva en el kernel de otra sesión. Una ejecución limpia producía `NameError`.

**Corrección:** definida en la celda de parámetros junto a `BASE_PATH`, del que ahora derivan todas las rutas.

### 3.2 Instalación de dependencias comentada — NB01

`%pip install pyreadstat boto3` estaba comentado pese a que el output almacenado demuestra que se ejecutó. Sin él, `import pyreadstat` falla.

**Corrección:** descomentado y con versión fijada (`pyreadstat==1.3.5`, la del output).

### 3.3 Bloque de depuración fallido con dependencia circular — NB02

Cuatro celdas entre los imports y la configuración leían `silver_delta` (que el propio notebook crea después), `gold_delta` (del NB03) y `scaler.pkl` + `svm_final.onnx` (del NB05). Sus outputs mostraban el fallo de forma explícita:

```
AUC del SVM: 0.5004  (esperado 0.9377)
✗ No coincide — revisar el one-hot o el orden de filas
```

Se verificó que era trabajo abandonado: los artefactos que pretendía generar (`scaler_real.json`, `medianas_reales.json`) no existen ni se referencian en ningún punto del repositorio. El problema se resolvió por otra vía en el NB03 (`scaler_correcto.json`).

**Corrección:** eliminadas las 4 celdas (86 líneas). Se comprobó previamente que ninguna de sus variables se usa después.

### 3.4 Celda final con `NameError` — NB03

`print(df_gold['LBXGH'].min())` sobre una variable inexistente. Era la última celda del notebook: lo último que veía quien lo abriera en GitHub.

**Corrección:** eliminada.

### 3.5 Escritura duplicada del CSV de features — NB03

Dos celdas de verificación temporal escribían `qsvm_top8_features.csv`, que la celda de guardado vuelve a escribir. Los comentarios delataban su origen (*"Verificación de seguridad antes de sobrescribir el CSV viejo"*).

**Corrección:** eliminadas las dos celdas.

### 3.6 Artefacto huérfano `X_train_qsvm.parquet` — NB03

Aparecía en el listado de validación del NB03 `(1500, 89)` y en la comprobación de leakage del NB02, pero ninguna celda lo escribía. El notebook ya calculaba `X_train_qsvm` sin persistirlo.

**Corrección:** añadida la escritura. El código pasa a coincidir con la salida de validación que ya lo listaba.

### 3.7 Valores SHAP sin productor — NB05

La celda 9 cargaba `shap_values_svm.npy` y `shap_X_test_sample_svm.npy` desde disco. La celda anterior **solo comprobaba que existían**. Ningún notebook los generaba: una ejecución limpia fallaba con `FileNotFoundError`.

**Corrección:** bloque de generación con `KernelExplainer` que se ejecuta únicamente si los ficheros faltan. La forma `(200, 89)` se dedujo inspeccionando los `.npy` existentes, no se inventó.

### 3.8 El NB05 no instalaba sus dependencias

Importaba `onnxruntime`, `skl2onnx` y `shap` con las tres líneas `%pip` comentadas. Solo funcionaba si se ejecutaba tras el NB04 en el mismo entorno. El único notebook que declara dependencias en su metadata de Databricks es el NB01.

**Corrección:** descomentadas las tres líneas.

### 3.9 Modo de ejecución del QSVM — NB06

Ver sección 5.1: el hallazgo más relevante de la auditoría.

### 3.10 Reinicio del kernel después de los imports — NB07

La celda `dbutils.library.restartPython()` estaba situada **después** de la celda de imports. Al descomentarla, habría borrado los imports recién realizados.

**Corrección:** reordenado a `pip → reinicio → imports`, con la numeración de rótulos ajustada.

---

## 4. Seguridad y privacidad

### 4.1 Credenciales AWS

Se verificó que **no había secretos reales** en ningún fichero ni en los 16 commits del historial (0 coincidencias con el patrón `AKIA/ASIA` + 16 caracteres). Tampoco se versionó nunca ningún `secrets.toml`, `.env`, `.pem` o fichero de credenciales.

El riesgo no era el placeholder, sino el flujo documentado (*"reemplaza los placeholders con tus credenciales reales"*), que implica que en algún momento el fichero en disco contiene la clave real.

**Corrección:** sustituido por `dbutils.secrets.get(scope="aws-nhanes", key=...)`. README y `TECHNICAL_NOTES` actualizados.

### 4.2 Dato personal en los outputs

El email del autor aparecía 14 veces en el NB02 y 8 en el NB01, dentro de la columna `userName` del historial de transacciones Delta.

**Corrección:** columna retirada del código y depurada del HTML y de la metadata de los outputs almacenados, preservando la integridad de las tablas.

> **Nota:** el email permanece en al menos 8 commits ya publicados del historial de git. Limpiarlo exigiría reescribir el historial (`git filter-repo`) y forzar push. Al tratarse del propio autor, cuyo nombre ya figura en el README, se recomienda no hacerlo.

### 4.3 Verificación del repositorio remoto

Remoto configurado: `https://github.com/juan-albornoz/tfm-qml-databricks-pipeline.git`, con 16 commits. **No se pudo determinar si es público o privado** — conviene confirmarlo antes de la entrega.

---

## 5. Hallazgos documentados sin corregir

Cuatro hallazgos exigirían modificar datos o modelos. Corregirlos invalidaría resultados ya obtenidos y defendidos, por lo que se documentaron en `TECHNICAL_NOTES.md`.

### 5.1 El modelo QSVM guardado no se puede recargar — `TECHNICAL_NOTES` 2.11

**El hallazgo más importante de la auditoría.** Los outputs del NB06 contenían un `TypeError` almacenado:

```
TypeError: ParameterExpression.__new__() missing 2 required positional
arguments: 'name_map' and 'expr'
```

`qsvm_final.pkl` **no se puede deserializar** con Qiskit 2.5.0, porque el pickle arrastra el `ZZFeatureMap` completo con sus objetos `ParameterExpression`. Es decir: **el atajo `TRAINING_MODE = False` que documentaban el README y las notas técnicas ("carga el modelo en ~2 min") no funciona.**

La prueba está en los propios outputs: las ramas `if TRAINING_MODE:` (entrenamiento, 21,1 min) y `if not TRAINING_MODE:` (carga de scores) tienen **ambas** resultados, siendo mutuamente excluyentes. Los outputs del NB06 provienen de dos sesiones distintas.

**Corrección aplicada (no altera resultados):** la carga va ahora en `try/except`; si el pickle no es legible, se informa del motivo y `TRAINING_MODE` pasa a `True` para reentrenar en lugar de abortar. Combinado con la autodetección del fichero, el notebook queda operativo en los tres escenarios posibles. Traceback obsoleto retirado.

### 5.2 `WTINT2YR` es una de las 89 features — `TECHNICAL_NOTES` 2.10

El peso de muestreo de la entrevista de NHANES **forma parte de los predictores** de los tres modelos.

Mecanismo: `GLU`, `INS` y `TRIGLY` traen `WTSAF2YR`, por lo que el merge del NB01 genera tres columnas (`WTSAF2YR`, `WTSAF2YR_x`, `WTSAF2YR_y`). La lista de exclusión solo casa con el nombre exacto. Las variantes `_x`/`_y` y `WTMEC2YR` las descarta el filtro de correlación, pero **`WTINT2YR` sobrevive**: el filtro elimina la segunda columna de cada par correlacionado, y en ese par le tocó a `WTMEC2YR`.

Verificado por simulación sobre los datos reales: se reproducen exactamente **89 features**, coincidiendo con el output del notebook, y `WTINT2YR` está entre ellas. Correlación con el objetivo: −0,1118. Rango: 4.363–137.870.

Un peso de muestreo codifica la probabilidad de selección en el diseño muestral, no una característica del paciente. No hay fuga del objetivo, pero sí es un artefacto del pipeline entre los predictores.

**No se corrige** porque excluirlo cambiaría el conjunto de features y, en cascada, los tres modelos y todas las métricas publicadas.

### 5.3 24 de 91 columnas de Silver tienen varianza cero — `TECHNICAL_NOTES` 2.8

La winsorización IQR×3 se aplica a todas las variables numéricas salvo cinco, lo que arrastra a variables categóricas codificadas (respuestas sí/no como 1/2, códigos 7 y 9). En una variable donde más del 75 % comparte valor, `Q1 = Q3` → `IQR = 0` → los límites se colapsan y `clip()` convierte la columna en constante.

Verificado empíricamente: 10 variables aplanadas por este mecanismo (`PAQ605`, `PAQ635`, `PAQ650`, `DMDCITZN`, `DMQMILIZ`, `SIALANG`, `FIALANG`, `MIALANG`, `AIALANGA`, `DMDHHSZA`), más 14 constantes por los filtros del pipeline. El propio notebook lo corrobora: `PAQ635` reporta 1.929 outliers corregidos, exactamente las filas no modales llevadas al valor modal.

**Confirmación independiente:** el NB03 reporta 23 columnas de varianza cero — las 24 de Silver menos `RIDSTATR`, que se excluye. Dos análisis por vías distintas dan el mismo número.

**Impacto:** ninguno sobre la validez de los resultados. Una constante aporta cero a la predicción; `StandardScaler` la convierte en ceros. Es pérdida de información potencialmente útil, no distorsión.

### 5.4 Filtro de correlación calculado antes del split — `TECHNICAL_NOTES` 2.9

Las 16 columnas descartadas por correlación r > 0.90 se determinan sobre el dataset completo, incluyendo las observaciones que después forman el test. Es una forma leve de fuga por selección de características.

**No afecta** al escalado (el `StandardScaler` se ajusta solo con train) ni a la selección de las 8 variables del QSVM (`RandomForestClassifier` entrenado solo con train). Ambos están correctamente resueltos.

---

## 6. Calidad de código

Aplicado transversalmente a los 7 notebooks:

- **Rutas centralizadas** en `BASE_PATH` por notebook, con comentarios que indican si cada ruta es de entrada, salida o solo lectura.
- **`except:` desnudos** sustituidos por excepciones concretas (`FileNotFoundError`, `OSError`, `AttributeError/TypeError/ValueError`).
- **10 `assert` de validación** que verifican invariantes del pipeline sin producir salida: integridad de la lectura S3, ausencia de duplicación en los joins, coherencia de filas y columnas entre cada capa y su tabla Delta, y control de pesos de muestreo.
- **Imports sin uso eliminados**: `os` y `seaborn` (NB01), `seaborn` y `matplotlib` (NB02 y NB03), `ConfusionMatrixDisplay` duplicado (NB04), `numpy` reimportado (NB05), `os` (NB07).
- **~20 f-strings sin interpolación** corregidas.
- **Lógica duplicada unificada**: `normalize_dtypes` se define una vez y `normalize_and_save` la reutiliza.
- **Constantes derivadas** en lugar de literales: `N_ARCHIVOS_ESPERADOS = len(ciclos) * len(modulos)` en lugar de `27`; número de features derivado en lugar de `89`.
- **Métricas leídas de sus CSV** en el gráfico comparativo del NB06, en lugar de estar fijas en el código. Se verificó previamente que los 12 valores coinciden exactamente, por lo que la figura no cambia.
- **Rótulos de celda** renumerados y desduplicados (dos "CELDA 15" en el NB06; encabezado con nivel inconsistente en el NB02).
- **Metadata**: `kernelspec` corregido de `.venv` a `Python 3.12 (Databricks Serverless)`, `language_info` completado, `execution_count` secuencial.

---

## 7. Documentación

### `TECHNICAL_NOTES.md`

- **Numeración corregida.** El documento tenía dos secciones "4" y las subsecciones de *Decisiones de diseño* numeradas como 3.x. Ahora: 1 Entorno · 2 Limitaciones · 3 Componentes cuánticos · 4 Decisiones de diseño · 5 Reproducibilidad · 6 Despliegue.
- **Cuatro secciones nuevas**: 2.8 winsorización, 2.9 filtro de correlación, 2.10 pesos de muestreo, 2.11 serialización del QSVM.
- **Sección 6 nueva**: despliegue de Streamlit, con la tabla de los 8 artefactos y su origen.
- Paso 3 de reproducibilidad actualizado al flujo de Databricks Secrets.

### `README.md`

- **Ciclos corregidos**: decía *2017-2018, 2019-2020, 2021-2022*; el pipeline usa *2013-2014 (\_H), 2015-2016 (\_I), 2017-2018 (\_J)*.
- Sección de credenciales reescrita para Databricks Secrets.
- Sección de modo de ejecución del QSVM actualizada, con aviso sobre la incompatibilidad de Qiskit.
- Sección nueva de despliegue de Streamlit.

### `requirements.txt`

- **`pyreadstat==1.3.5` añadido** — faltaba, pese a ser la librería con la que el NB01 lee los 27 XPT.
- Nota aclarando que `pyspark` y `delta-spark` los aporta el runtime de Databricks.

### Configuración local del IDE

`.vscode/settings.json` creado en dos ubicaciones para silenciar los avisos de Pylance sobre `pyspark`, `delta.tables` y `dbutils`, que el runtime de Databricks inyecta y no existen localmente. **Excluido por `.gitignore`**, no se publica.

---

## 8. PENDIENTE — Plan de prueba

**Nada de esto se ha ejecutado.** Orden sugerido, de mayor a menor riesgo:

| # | Paso | Qué vigilar |
|-|-|-|
| 1 | **NB01, parar en la celda 6** | `dbutils.secrets.get()`. Si el scope `aws-nhanes` no existe, falla. Si la cuenta es Community Edition clásica, la API de secretos puede no estar disponible |
| 2 | **NB01 completo** | 4 asserts nuevos: lectura de los 27 XPT, duplicación en joins, coherencia Bronze ↔ Delta |
| 3 | **NB02 completo** | 3 asserts; verificación de leakage con el nuevo mensaje |
| 4 | **NB03 completo** | `assert set(pesos) <= {"WTINT2YR"}` y la escritura nueva de `X_train_qsvm.parquet` |
| 5 | **NB04 completo** | El menos modificado; sirve de control |
| 6 | **NB05 completo** | Los `%pip` descomentados pueden resolver versiones distintas de `skl2onnx`/`onnx`. El bloque SHAP debe **saltarse** e imprimir solo las dos líneas de verificación |
| 7 | **NB06 completo** | Debe imprimir `TRAINING_MODE: False`. Si el pickle sigue sin deserializarse, lo indicará y pasará a reentrenar (**3 horas**) |
| 8 | **NB07 completo** | Orden de celdas modificado |

### Riesgos concretos del código nuevo

| Riesgo | Ubicación | Comentario |
|-|-|-|
| **Alto** | `dbutils.secrets.get()` — NB01 | Depende de infraestructura que puede no existir |
| **Alto** | Bloque SHAP — NB05 celda 9 | ~12 líneas nuevas nunca ejecutadas. En el entorno actual no se ejecutarán porque los `.npy` existen |
| **Medio** | `%pip` descomentados — NB05 | Resolución de versiones |
| **Medio** | `try/except` del QSVM — NB06 | Cambio de comportamiento: puede disparar 3 h de reentrenamiento |
| **Bajo** | `X_train_qsvm.to_parquet` — NB03 | Línea nueva sobre una variable ya existente |
| **Bajo** | Reordenado — NB07 | Las celdas movidas están comentadas |

> **Un `assert` que introduje ya rompió el NB03 y fue detectado y corregido** antes de esta entrega, precisamente por hacer esta comprobación. Es la mejor evidencia de por qué hay que ejecutar el pipeline antes de publicar.

---

## 9. PENDIENTE — Decisiones del autor

1. **Ejecutar el pipeline completo** y comunicar cualquier traceback.
2. **Confirmar si el repositorio de GitHub es público o privado.**
3. **Subir los cambios a GitHub** (todo el trabajo está solo en local).
4. **Decidir sobre el email en el historial de git**: se recomienda no reescribirlo.
5. **Valorar añadir `*.XPT` al `.gitignore`** — actualmente solo ignora `*.xpt` en minúscula, y los ficheros NHANES son mayúsculas. Preventivo: no hay ningún XPT en el repositorio.
6. **Considerar reejecutar el NB02 y el NB06** para unificar outputs:
   - NB02: el filtro DIQ010 muestra `7831 → 7831 (0 eliminados)`, cuando en una ejecución lineal sería `7835 → 7831 (4 eliminados)`. Los datos finales son correctos; solo el mensaje es engañoso.
   - NB06: outputs procedentes de dos sesiones con distinto `TRAINING_MODE`.
7. **Líneas de trabajo futuro** identificadas para la memoria: corregir la winsorización de categóricas (5.3), excluir `WTINT2YR` (5.2) y re-evaluar los tres modelos.

---

## 10. Correcciones a afirmaciones propias

Por transparencia, dos afirmaciones erróneas emitidas durante la auditoría y rectificadas:

1. **Sufijos de merge (NB01).** Se afirmó que no existían columnas `_x`/`_y` basándose en que las sumas de columnas cuadraban. El razonamiento era inválido: una colisión de nombres no cambia el número de columnas, solo las renombra. La comprobación posterior sobre los datos reveló tres columnas `WTSAF2YR`, lo que condujo al hallazgo 5.2.

2. **Etiquetas compartidas en la app.** Se afirmó que un cambio de partición produciría curvas ROC mal calculadas sin aviso. Es inexacto: `app.py:679` ya comprueba que las longitudes de scores y etiquetas coincidan y devuelve `None` en caso contrario. El riesgo residual se limita a una partición del mismo tamaño y distinta composición.

Adicionalmente, se afirmó que eliminar las celdas 7–10 del NB02 rompía su dependencia circular con el NB03. Solo fue parcial: la verificación de leakage sigue leyendo los parquet de Gold. Se documentó en el encabezado de la celda en lugar de reubicarla, para no perder su salida.

---

## 11. Anexos

### Copias de seguridad

`C:\Users\ingel\AppData\Local\Temp\claude\...\scratchpad\backup_full\` — los 7 notebooks, `app.py`, `README.md` y `TECHNICAL_NOTES.md` en su estado previo a la auditoría.

> Es un directorio temporal de sesión. **Cópialo a una ubicación permanente si quieres conservarlo.**

### Documento de trazabilidad

Mapa visual de dependencias entre notebooks y artefactos, con el estado de cada eslabón:
`https://claude.ai/code/artifact/66224f1d-dc1e-4cb1-871e-624dec1d24fd`

### Contrato de artefactos — no modificar sin revisar toda la cadena

| Artefacto | Consumido por |
|-|-|
| Nombres de fichero en `gold/` | NB04, NB05, NB06 los leen por nombre literal |
| `random_state=42`, `test_size=0.20` | Define el test compartido por los tres modelos y la app |
| Orden de `qsvm_top8_features.csv` | El NB06 mapea las 8 columnas por posición |
| Claves de `scaler_correcto.json` | `app.py` lee `features`, `mean`, `scale` |
| Patrón `{prefix}_y_scores.npy` | `app.py` compone el nombre |
| Columna `SPLIT` en `gold_delta` | Única forma de separar train/test en la capa Delta |

### Ficheros modificados

```
 M README.md
 M TECHNICAL_NOTES.md
 M requirements.txt
 M notebooks/notebook_01_bronze.ipynb
 M notebooks/notebook_02_silver.ipynb
 M notebooks/notebook_03_gold.ipynb
 M notebooks/notebook_04_lgbm.ipynb
 M notebooks/notebook_05_svm.ipynb
 M notebooks/notebook_06_qsvm.ipynb
 M notebooks/notebook_07_validacion_silver.ipynb
```

No versionado (excluido por `.gitignore`): `.vscode/settings.json` en la raíz del workspace y del repositorio.

---

*Informe generado el 3 de agosto de 2026. Auditoría por análisis estático, sin ejecución del pipeline.*
