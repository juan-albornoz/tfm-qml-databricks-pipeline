# TFM — Quantum Machine Learning en pipeline DataOps sobre Databricks

**Máster en Análisis de Datos Masivos (Big Data)**  
Universidad Europea de Valencia — Curso 2025-2026  
**Autor:** Juan Albornoz Carrasco  
**Director:** Prof. Ronal Muresano

\---

## Descripción

Este repositorio contiene la implementación completa del Trabajo Fin de Máster:

> \*"Integración de Quantum Machine Learning en un pipeline DataOps: arquitectura Medallón sobre Databricks y comparativa con modelos clásicos"\*

El proyecto diseña e implementa un framework DataOps end-to-end sobre **Databricks Community Edition**, con **AWS S3** como capa de almacenamiento cloud real y una **arquitectura Medallón** (Bronze → Silver → Gold) sobre **Delta Lake**. El componente diferencial es la integración de un **Quantum Support Vector Machine (QSVM)** con `ZZFeatureMap` y `FidelityQuantumKernel` (Qiskit), comparado experimentalmente contra modelos clásicos de referencia bajo condiciones controladas.

\---

## Stack tecnológico

|Capa|Tecnología|
|-|-|
|Almacenamiento cloud|AWS S3 (`tfm-nhanes`, región `eu-west-1`)|
|Plataforma de datos|Databricks Community Edition (Serverless)|
|Arquitectura de datos|Delta Lake — Medallón (Bronze / Silver / Gold)|
|ML clásico|LightGBM 4.x, SVM-RBF (scikit-learn)|
|Quantum ML|Qiskit, ZZFeatureMap, FidelityQuantumKernel|
|Interpretabilidad|SHAP (TreeExplainer)|
|Serialización|ONNX (LightGBM, SVM-RBF)|
|Calidad del dato|dataframe-expectations 0.7.0|
|Despliegue|Streamlit Community Cloud|

\---

## Dataset

**NHANES** (National Health and Nutrition Examination Survey) — CDC, EE.UU.

* 3 ciclos: 2013-2014 (`\_H`), 2015-2016 (`\_I`), 2017-2018 (`\_J`)
* 27 archivos XPT — 9 módulos por ciclo
* 29.400 registros Bronze → 7.831 registros Silver
* **Variable objetivo:** `DIQ010` binarizada (diabetes tipo 2)
* Desbalance de clases: 86% negativo / 14% positivo

> ⚠️ Los datos NHANES no se incluyen en este repositorio por restricciones de licencia CDC.  
> Descarga disponible en: https://wwwn.cdc.gov/nchs/nhanes/

\---

## Estructura del repositorio

```
tfm-qml-databricks-pipeline/
├── notebooks/
│   ├── notebook\_01\_bronze.ipynb              # Ingesta ELT desde AWS S3 → Bronze
│   ├── notebook\_02\_silver.ipynb              # Limpieza y transformación → Silver
│   ├── notebook\_03\_gold.ipynb                # Preparación para modelado → Gold
│   ├── notebook\_04\_lgbm.ipynb                # Modelo LightGBM + SHAP + ONNX
│   ├── notebook\_05\_svm.ipynb                 # Modelo SVM-RBF + ONNX
│   ├── notebook\_06\_qsvm.ipynb                # Modelo QSVM (ZZFeatureMap + FidelityQuantumKernel)
│   └── notebook\_07\_validacion\_silver.ipynb   # Suite dataframe-expectations
├── streamlit/
│   └── app.py                                # Aplicación de resultados interactiva
├── requirements.txt
├── .gitignore
└── LICENSE
```

\---

## Pipeline DataOps

```
AWS S3 (27 archivos XPT)
    │
    ▼  boto3
┌─────────────────────────────────────────┐
│  Bronze — Delta Lake                    │
│  29.400 registros · 162 columnas        │
└─────────────────────────────────────────┘
    │  filtros · binarización · winsorización · imputación
    ▼
┌─────────────────────────────────────────┐
│  Silver — Delta Lake                    │
│  7.831 registros · 91 columnas          │
└─────────────────────────────────────────┘
    │  encoding · correlación · split 80/20 · StandardScaler
    ▼
┌─────────────────────────────────────────┐
│  Gold — Delta Lake                      │
│  6.264 train · 1.567 test · 89 features │
└─────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────────────────┐
    ▼                  ▼                              ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────────────────┐
│ LightGBM │    │   SVM-RBF    │    │           QSVM              │
│GridSearch│    │  C=1.0       │    │  ZZFeatureMap (8q, reps=2)  │
│SHAP+ONNX │    │  ONNX        │    │  FidelityQuantumKernel      │
└──────────┘    └──────────────┘    │  500 instancias train       │
                                    └─────────────────────────────┘
    │
    ▼
Streamlit Community Cloud
```

\---

## Resultados experimentales

Comparativa triangulada sobre el mismo conjunto de test (1.567 instancias):

|Modelo|AUC-ROC|F1-macro|Accuracy|MCC|Mejor en|
|-|-|-|-|-|-|
|**LightGBM**|**0.9485**|0.6523|0.7243|0.4566|AUC-ROC|
|**SVM-RBF**|0.9377|**0.8243**|**0.9075**|**0.6539**|F1, Accuracy, MCC|
|**QSVM**|0.5493|0.4669|0.8602|0.0625|—|

> QSVM entrenado sobre muestra estratificada de 500 instancias por coste computacional O(n²).  
> Los tres modelos se evalúan sobre el mismo conjunto de test completo para comparabilidad.

\---

## Ejecución

### Requisitos previos

* Databricks Community Edition con Unity Catalog habilitado
* Bucket AWS S3 con los 27 archivos XPT de NHANES
* Credenciales AWS IAM con permisos de lectura sobre el bucket

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Orden de ejecución de los notebooks

```
1. notebook\_01\_bronze.ipynb          # \~5 min
2. notebook\_02\_silver.ipynb          # \~3 min
3. notebook\_03\_gold.ipynb            # \~2 min
4. notebook\_04\_lgbm.ipynb            # \~10 min (GridSearchCV 5-fold)
5. notebook\_05\_svm.ipynb             # \~15 min
6. notebook\_06\_qsvm.ipynb            # \~22 min entrenamiento + \~132 min predicción
7. notebook\_07\_validacion\_silver.ipynb  # \~1 min
```

### Configuración de credenciales AWS

Las credenciales se gestionan con **Databricks Secrets**: nunca se escriben en el notebook. Alta del scope (una sola vez, desde la CLI de Databricks):

```bash
databricks secrets create-scope --scope aws-nhanes
databricks secrets put-secret   --scope aws-nhanes --key access_key
databricks secrets put-secret   --scope aws-nhanes --key secret_key
```

`notebook_01_bronze.ipynb` las recupera en tiempo de ejecución:

```python
access_key = dbutils.secrets.get(scope="aws-nhanes", key="access_key")
secret_key = dbutils.secrets.get(scope="aws-nhanes", key="secret_key")
```

> ⚠️ Nunca escribas credenciales en claro en los notebooks ni las subas al repositorio.

### Modo ejecución QSVM

El notebook QSVM detecta automáticamente si ya existe un modelo entrenado, para evitar repetir las ~3 horas de cómputo:

```python
TRAINING_MODE = not os.path.exists(f"{models_dir}/qsvm_final.pkl")
# sin qsvm_final.pkl -> True  : entrena desde cero (~154 min en total)
# con qsvm_final.pkl -> False : carga el modelo guardado (~2 min en total)
```

Puede forzarse a `True` manualmente si se desea re-entrenar sobre un modelo ya existente.

> ⚠️ Si el entorno ha actualizado Qiskit desde que se entrenó el modelo, el `.pkl` no puede deserializarse. El notebook detecta el fallo y re-entrena automáticamente — ver [TECHNICAL\_NOTES.md](TECHNICAL_NOTES.md), sección 2.11.

### Despliegue de la aplicación Streamlit

Streamlit Community Cloud solo accede al contenido del repositorio, no a Unity Catalog Volumes. Tras ejecutar los notebooks de modelado hay que copiar a `streamlit/models/` los 8 artefactos que consume la app: `lgbm_final.onnx`, `svm_final.onnx`, `scaler_correcto.json`, `medianas_correctas.json`, `lgbm_y_scores.npy`, `svm_y_scores.npy`, `qsvm_y_scores.npy` y `qsvm_y_test.npy`.

Detalle completo en [TECHNICAL\_NOTES.md](TECHNICAL_NOTES.md), sección 6.

\---

## Versiones del entorno

|Librería|Versión|
|-|-|
|Python|3.12|
|pandas|1.5.3|
|numpy|1.23.5|
|scikit-learn|1.6.1|
|lightgbm|4.x|
|qiskit|2.5.0|
|qiskit-machine-learning|0.9.0|
|qiskit-algorithms|0.4.0|

\---

## Licencia

MIT License — ver [LICENSE](LICENSE)

\---

*Trabajo Fin de Máster — Universidad Europea de Valencia — 2025-2026*

