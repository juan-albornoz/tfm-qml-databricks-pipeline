# Informe de auditoría — Código y comportamiento en ejecución

**Fecha:** 17 de agosto de 2026 · **Alcance:** `streamlit/app.py` (6.774 líneas), `streamlit/i18n.py`
(4.109 líneas), los artefactos de `streamlit/models/` y la aplicación **en marcha** en las siete
páginas, los dos temas y los cinco idiomas.

Complementa a [INFORME_AUDITORIA_DASHBOARD.md](INFORME_AUDITORIA_DASHBOARD.md), que auditó el
**comportamiento numérico**. Aquel preguntaba si las cifras son ciertas; este pregunta si el código
que las pinta hace lo que dice, si algo falla en ejecución y si algo sobra.

---

## 0. Resumen ejecutivo

**No hay ningún fallo que rompa la aplicación.** Las siete páginas se pintan sin una sola excepción
de Python en los dos temas, la reconciliación de métricas cierra, la inferencia ONNX reproduce los
modelos y el catálogo de los cinco idiomas está completo y cuadrado.

Se encontraron **cinco defectos**, todos corregidos y verificados:

| # | Defecto | Gravedad | Dónde |
|-|-|-|-|
| 1 | El perfil de referencia del Predictor es un paciente **imposible**: varón · blanco no hispano **y** asiático no hispano a la vez | **Alto** | `_perfil_base` |
| 2 | En tema oscuro, las etiquetas `q₀ \|0⟩` del circuito de 3 qubits son **ilegibles** (1,10:1) | **Alto** | `.fig-card` · Esfera de Bloch |
| 3 | Los controles de la Esfera de Bloch **pierden su valor** al cambiar de idioma | Medio | `st.selectbox` / `st.slider` sin `key` |
| 4 | `height="auto"` no es válido en SVG: **error de consola** en cada pulsación de puerta | Bajo | `ent_circuito_svg` |
| 5 | Curvas ROC **suavizadas con spline** bajo un subtítulo que promete "punto a punto" | Bajo | página Resultados |

Más una **retirada de código muerto** (las tres reglas CSS de `.gov-dim`, que ya no casan con nada).

---

## 1. Lo que se verificó y está correcto

### 1.1 Datos y métricas

Recalculado desde los `.npy` embarcados, cada modelo en **su propio umbral**:

| Modelo | AUC declarado | AUC recalculado | Matriz | accuracy · F1-macro · MCC |
|-|-|-|-|-|
| LightGBM | 0,9485 | **0,9485** ✓ | exacta ✓ | los tres ✓ |
| SVM-RBF | 0,9377 | **0,9377** ✓ | exacta ✓ | los tres ✓ |
| QSVM | 0,5493 | **0,5493** ✓ | exacta ✓ | los tres ✓ |

Test: 1.567 instancias, 220 positivas (14,04 %). El sello **✓ Reconciliadas** que muestra la página
es cierto.

### 1.2 Artefactos

- `scaler_correcto.json`: 89 features · 89 medias · 89 escalas, alineadas con `medianas_correctas.json`
  sin sobrantes ni ausentes, ninguna escala a cero, y las 8 variables del QSVM presentes.
- Las dos sesiones ONNX cargan y responden. El segundo tensor de salida llega con **forma distinta**
  en cada modelo (LightGBM `ndarray (N,2)`, SVM `list` de dicts) y `out[1][i][1]` acierta en ambos,
  como documenta el código.
- `golden_lgbm.npz` / `golden_svm.npz` **no están en el repositorio**, y la aplicación lo dice: el
  estado es *«Sin verificar»* y nunca afirma haber comprobado lo que no ha comprobado.
- `validacion_silver_dfe.csv` tampoco está, y la suite cae a los valores publicados sin incidencia.

### 1.3 Internacionalización

Los cinco catálogos tienen **249 claves exactas**, ninguna falta ni sobra, y —comprobado
estructuralmente— **la misma forma en todos**: mismo tipo, misma longitud de lista y los **mismos
marcadores de `format()`**. Ninguna clave usada por `S()` carece de entrada. El buscador devuelve
resultados coherentes en los cinco idiomas.

### 1.4 Comportamiento en ejecución

Recorridas las 7 páginas × 2 temas con navegador dirigido por CDP: **0 excepciones de Python, 0
errores de consola** (tras las correcciones) y sin desbordamiento horizontal en ninguna. Verificados
además, uno a uno:

- los 8 deslizadores del Predictor mueven la predicción y las dos tarjetas de modelo;
- la curva de respuesta cambia de eje y de datos con las 8 variables del selector;
- la secuencia de puertas da la lectura esperada del GHZ — |r| 1 → 0 → 0, concurrencia 0 → 1 → **0**;
- la medición muestrea correctamente (494/506 en `|000⟩`/`|111⟩`, cero en las otras seis);
- las 3 pestañas de Gobernanza, las 2 de SHAP y `?tab=` tras recargar;
- las 5 banderas, el interruptor de tema, el colapso de la barra y el buscador con sus tres destinos.

Los **980 selectores** de la hoja de estilos se probaron contra el DOM real de las siete páginas: no
hay ninguno inválido, y de los que no casan con nada solo `.gov-dim` era código propio muerto (el
resto son pseudoclases, estados transitorios y hojas del propio Streamlit).

---

## 2. Defectos encontrados y corregidos

### 2.1 [ALTO] El paciente de referencia sigue siendo imposible

`_perfil_base` rellena con su moda todo grupo *one-hot* que sume cero. La regla es correcta **solo
si el grupo está completo**, y dos de los cinco no lo están:

| Grupo | Columnas | Σ prevalencias | P(ninguna) | ¿completo? |
|-|-|-|-|-|
| RIDRETH1 | 5 | 1,000 | 0,000 | sí |
| DMDEDUC2 | 6 | 1,000 | 0,000 | sí |
| DMDMARTL | 7 | 1,000 | 0,000 | sí |
| **RIAGENDR** | 1 | 0,485 | **0,515** | **no** |
| **RIDRETH3** | 2 | 0,167 | **0,833** | **no** |

`RIAGENDR` viene codificado dejando una categoría fuera (solo queda `RIAGENDR_1.0`, varón), y de
`RIDRETH3` sobrevivieron al filtro de correlación únicamente las dos categorías que `RIDRETH1` no
tiene. En los dos casos **el cero es una categoría, y además la más frecuente**: mujer (51,5 %) y
"ni asiático ni otro" (83,3 %).

Rellenándolos a ciegas, el perfil quedaba en varón —cuando la moda es mujer— y **blanco no hispano
por `RIDRETH1` y asiático no hispano por `RIDRETH3` a la vez**, una combinación que no existe en el
conjunto de datos. Es exactamente el defecto que §1.9 del informe anterior tituló *«El paciente base
es imposible»*: la corrección C3 se quedó a mitad de camino.

**Corregido** comparando la moda explícita contra la categoría implícita, cuya prevalencia es 1 − Σ.
Gana la más probable. En los tres grupos completos no cambia nada; en los dos incompletos el vector
se queda a ceros.

Impacto medido sobre el perfil por defecto: **LightGBM no se mueve** (no ramifica sobre estas
dummies) y **SVM-RBF baja de 2,53 % a 2,34 %**. En la zona de decisión pesa más: con HbA1c 6,5 el
SVM pasa de 29,7 % a **35,9 %**.

### 2.2 [ALTO] Etiquetas ilegibles en el circuito de 3 qubits, en tema oscuro

`.fig-card` fija `background:#FFFFFF` porque está pensada para las **láminas raster** —el beeswarm de
SHAP y el circuito de 8 qubits— que traen ese fondo dentro del PNG. Pero el circuito de 3 qubits de
la Esfera de Bloch **no es una lámina**: es un SVG que se dibuja en cada pulsación con los tokens de
la paleta activa. Sobre el blanco fijo, la tinta del rótulo en tema oscuro es la niebla `#F1F5F9`:
**1,10:1**, es decir, invisible. Los tres `q₀ |0⟩` no se leían.

**Corregido** con una variante `.fig-card.fig-vector` que le devuelve la superficie del tema. El
rótulo pasa a 16,1:1 y la tarjeta iguala a la de la Q-sphere que tiene al lado. **En tema claro no
cambia ni un píxel** (allí `t['surface']` ya es `#FFFFFF`). Las láminas raster conservan su blanco.

### 2.3 [MEDIO] Los controles de la Esfera de Bloch se rebobinan al cambiar de idioma

El selector de variable y sus dos deslizadores no llevaban `key`, y para Streamlit la identidad de un
widget sin clave **es su rótulo** — que se traduce. Medido en el navegador: con la glucosa a 6,9,
pulsar una bandera devolvía a HbA1c y a 5,7. Los ocho deslizadores del Predictor, que sí tienen clave,
conservaban el suyo: la misma incoherencia que `tabs_i18n()` ya había resuelto para las pestañas,
en la única página donde quedaba.

**Corregido** con `key="bl_var"` (guarda el código NHANES, que no se traduce), `key=f"bl_val_{code}"`
(el rango y el paso son propios de cada variable) y `key="ent_shots"`. Verificado: el valor sobrevive
al cambio de bandera, de tema y al colapso de la barra.

### 2.4 [BAJO] `height="auto"` no es un valor válido de atributo SVG

`ent_circuito_svg` emitía `<svg width="100%" height="auto">`. `auto` solo existe como valor **CSS**;
como atributo de presentación el navegador lo rechaza y anota `Error: <svg> attribute height:
Expected length, "auto"` en consola **cada vez que se pinta el circuito**, o sea en cada pulsación de
puerta. **Corregido** omitiendo el atributo —el alto lo deduce el `viewBox`— y dejando `height:auto`
en el `style`, que ahí sí es CSS. La consola de la página queda limpia.

### 2.5 [BAJO] Curvas ROC suavizadas (§2.5 del informe anterior, sin aplicar)

Las tres curvas llevaban `shape="spline", smoothing=0.4` justo debajo de un subtítulo que promete
*«curvas empíricas reales, punto a punto»*. Una ROC empírica es una escalera —un peldaño por
instancia—, y suavizarla dibuja una continuidad que los datos no tienen: el mismo error cosmético que
la curva de respuesta del Predictor evita a propósito. **Corregido**: poligonal literal.

### 2.6 Código muerto retirado

Las tres reglas de `.gov-dim` (cabecera de dimensión de la suite de calidad) ya no las lleva ningún
elemento: al plegar las 15 expectativas en un expander por dimensión, ese rótulo pasó a ser el título
del widget. Comprobado contra el DOM: **0 coincidencias**. Retiradas, con nota de por qué.

---

## 3. Hallazgos que se dejan abiertos, y por qué

Ninguno rompe nada. Se documentan aquí en vez de tocarlos porque son **decisiones del autor**, no
defectos, y dos de ellos ya estaban valorados en el informe anterior y quedaron deliberadamente fuera
de los grupos aplicados.

| # | Hallazgo | Por qué no se toca |
|-|-|-|
| A | **`.onnx` de LightGBM con opset 1** en el dominio por defecto (`onnxruntime` avisa al arrancar) | Es el pendiente **C4**: exige reejecutar los notebooks 04/05 en Databricks. Hoy los resultados son correctos (100 % contra el pickle) |
| B | **"15/15 artefactos sin leakage"** (§3.4) | Texto visible del TFM en cinco idiomas. El informe anterior lo clasificó BAJO y sugirió *«sin fuga de etiqueta»*; la decisión es del autor |
| C | **Nota SHAP del SVM** (§4.1): dice que el ranking "coincide" sin matizar que el orden difiere (edad 2ª vs 4ª) y que las escalas no son comparables | Igual que B: prosa de la memoria, en cinco idiomas |
| D | **"Pass rate de la suite: 1,0"** | Correcto como tasa. `100 %` se leería antes, pero la columna del CSV es literalmente `pass_rate` |
| E | `Invalid color passed for textColor in theme.sidebar` en consola | Aviso interno de Streamlit 1.55, sin efecto visual. Fijarlo en `config.toml` rompería uno de los dos temas, porque ese ajuste es del servidor |

---

## 4. Reproducibilidad

Los recálculos de §1.1 y §2.1 se hicieron con el mismo `onnxruntime` que usa la aplicación, y los tres
scripts del informe anterior siguen sirviendo:

```bash
python audit_scripts/audit_metrics.py     # AUC, matrices y métricas desde los .npy
python audit_scripts/audit_baseline.py    # vector base y one-hot
python audit_scripts/audit_sens.py        # barridos de sensibilidad
```

La verificación en ejecución se hizo dirigiendo Chrome sin interfaz por CDP contra el servidor local,
recorriendo las siete páginas en los dos temas y accionando cada control. No requiere dependencias
nuevas del proyecto.
