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

* 3 ciclos: 2017-2018, 2019-2020, 2021-2022
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

En `notebook\_01\_bronze.ipynb`, reemplaza los placeholders con tus credenciales reales:

```python
access\_key = "TU\_ACCESS\_KEY\_ID"      # Reemplazar antes de ejecutar
secret\_key = "TU\_SECRET\_ACCESS\_KEY"  # Reemplazar antes de ejecutar
```

> ⚠️ Nunca subas credenciales reales al repositorio.

### Modo ejecución QSVM

El notebook QSVM soporta dos modos para evitar re-entrenar (\~3 horas):

```python
TRAINING\_MODE = True   # Entrena desde cero (\~154 min total)
TRAINING\_MODE = False  # Carga modelo guardado (\~2 min total)
```

\---

## Versiones del entorno

|Librería|Versión|
|-|-|
|Python|3.12|
|pandas|1.5.3|
|numpy|1.23.5|
|scikit-learn|1.6.1|
|lightgbm|4.x|
|qiskit|2.4.1|
|qiskit-machine-learning|0.9.0|
|qiskit-algorithms|0.4.0|

\---

## Licencia

MIT License — ver [LICENSE](LICENSE)

\---

*Trabajo Fin de Máster — Universidad Europea de Valencia — 2025-2026*

