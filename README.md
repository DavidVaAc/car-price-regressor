# 🚗 Car Price Regressor: Valuación Automatizada de Vehículos Usados ⚙️

Pipeline completo de Machine Learning y aplicación interactiva en Streamlit para automatizar la valuación de vehículos de segunda mano. El proyecto compara seis algoritmos de regresión bajo criterios de **precisión, latencia y coste de reentrenamiento**, selecciona el ganador (LightGBM) y lo expone como un servicio web con filtros en cascada que garantizan combinaciones de vehículo realistas.

---

## 📊 Resumen Ejecutivo

* **🎯 Objetivo de negocio:** RMSE estricto **< 2,500 €** sobre conjunto ciego, latencia de predicción suficiente para uso en tiempo real y reentrenamiento ágil sobre la infraestructura del operador.

* **🏆 Modelo seleccionado — LightGBM:**

    | Métrica | Valor | Lectura operativa |
    |---|---|---|
    | RMSE (test ciego) | **~1,592 €** | ~908 € por debajo del umbral; <15 % del precio promedio del dataset. |
    | R² | **~0.87** | Alta capacidad de generalización sin sobreajuste. |
    | Latencia de predicción | **~0.09 s** | Respuesta instantánea para la app. |
    | Tiempo de reentrenamiento | **~1.8 s** | Refresh del modelo a coste despreciable. |
    | Investigación (R&D, GridSearch CV=5) | **~39 s** | Iteración experimental rápida. |

* **🥊 Benchmark frente a 5 alternativas:** los tres boosters (LightGBM, XGBoost, CatBoost) lideran en precisión; LightGBM gana por **mejor relación RMSE / latencia / tiempo de entrenamiento**. La Regresión Lineal sirve de *sanity check* y confirma que la relación precio↔características **no es lineal** (RMSE > 2,800 €).

* **🔍 Interpretabilidad (SHAP):** la triada **`registration_year` × `power` × `mileage`** concentra la mayor parte del poder predictivo, seguida de `model`, `brand` y `not_repaired`. El comportamiento del modelo coincide con la lógica del mercado de segunda mano.

<p align="center">
  <img src="images/depreciacion_mercado.png" width="600" alt="Curva de depreciación capturada por el modelo">
</p>

* **🛠️ Filtros en cascada en la UI:** la aplicación encadena los selectores (`brand` → `model` → resto de características) sobre el histórico real, impidiendo que el usuario forme combinaciones inexistentes (ej. modelo que nunca ha tenido caja automática) y eliminando predicciones sobre datos imposibles.

<p align="center">
  <img src="images/app_screenshot.png" width="600" alt="Captura de la app de Streamlit">
</p>

---

## 🛠️ Acceso al Proyecto

### 📊 [Aplicación Interactiva (Streamlit)](https://car-price-regressor.streamlit.app/)
> Cotización en euros en tiempo real con filtros dinámicos. Ideal para una evaluación rápida del producto final.

### 📓 [Notebook de Modelado (Jupyter)](https://github.com/DavidVaAc/car-price-regressor/blob/main/notebooks/car_price_regressor.ipynb)
> Documentación técnica completa: EDA, tratamiento de outliers físicos, ingeniería de características, *grid search* sobre seis algoritmos, benchmark de producción y análisis SHAP.

---

## 🔬 Metodología y Decisiones de Ingeniería

### 1. ✅ Auditoría y Curación de Datos

El dataset, proveniente de un *scraping* del mercado alemán, contenía ruido estructural significativo. Se aplicaron varias capas de saneamiento:

* **Filtros físicos:** descarte de precios = 0, años de registro fuera del rango plausible (anteriores a 1910 o posteriores a 2016) y potencias imposibles (0 CV o > 1,000 CV).
* **Imputación localizada por `brand` + `model`:** los nulos en `vehicle_type`, `gearbox`, `fuel_type`, `not_repaired` y `power` se rellenan con la moda/mediana del **mismo modelo**, recuperando miles de registros sin distorsionar la distribución global.
* **Validación post-limpieza por promedios agrupados:** un segundo barrido detectó dos *buckets centinela* del formulario original que pasaban inadvertidos en el `pairplot`:
    * `registration_year == 2016` (año de borde mal poblado, mezcla coches genuinos con valores por defecto del scraper).
    * `mileage == 5000` (valor mínimo del selector, contaminado por usuarios que no informaron el kilometraje real).

    Ambos buckets se eliminaron, dejando las relaciones marginales precio↔año y precio↔kilometraje monótonamente coherentes con la lógica del mercado.

* **Optimización de memoria y E/S:** persistencia del dataset limpio en **Apache Parquet** y casting explícito a `category` para las variables cualitativas, lo que reduce el peso en disco y acelera la carga en producción.

### 2. 🤖 Modelado y Benchmark

Seis algoritmos compitieron bajo idénticas condiciones (`GridSearchCV`, 5 *folds*, *split* 80/20), con preprocesamiento ajustado a cada arquitectura:

* `LinearRegression` — *sanity check* con `OneHotEncoder` + `StandardScaler` + `TargetEncoder` (smoothing afinado).
* `DecisionTreeRegressor`, `RandomForestRegressor` — *Target Encoding* sobre `model`/`brand` para evitar la dispersión del OHE.
* `XGBRegressor` — `tree_method='hist'` y *Target Encoding* tuneado simultáneamente con la profundidad del árbol.
* `CatBoostRegressor` — `cat_features` nativo + `grid_search` interno.
* `LGBMRegressor` — categóricas en formato `category` consumidas nativamente.

| Modelo | RMSE (CV) | Tiempo R&D | `train_time` | `predict_time` | Veredicto |
|---|---|---|---|---|---|
| 🥇 **LGBMRegressor** | **~1,592 €** | **~39.4 s** | **~1.8 s** | **~0.09 s** | **✅ Seleccionado** |
| 🥈 XGBRegressor | ~1,594 € | ~56.7 s | ~3.3 s | ~0.26 s | ✅ Cumple |
| 🥉 CatBoostRegressor | ~1,634 € | ~251 s ⚠️ | ~12.2 s ⚠️ | ~0.05 s | ⚠️ Cumple |
| RandomForestRegressor | ~1,625 € | ~286 s ⚠️ | ~14.4 s ⚠️ | ~1.15 s ⚠️ | ⚠️ Cumple |
| DecisionTreeRegressor | ~1,879 € | ~59 s | ~0.8 s | ~0.05 s | ⚠️ Cumple |
| LinearRegression | > 2,800 € | ~3.6 s | ~0.5 s | ~0.06 s | ❌ Sanity check |

> 💡 Los tiempos se midieron en un entorno controlado y pueden variar en producción.

### 3. 🧠 Interpretabilidad (SHAP)

El análisis con `shap.TreeExplainer` sobre 1,000 observaciones de test confirma:

* **Ejes dominantes:** antigüedad (`registration_year`), potencia (`power`) y desgaste (`mileage`).
* **Identidad del vehículo:** `model` y `brand` actúan como multiplicadores de prima de mercado.
* **Estado:** `not_repaired` introduce un descuento claro y consistente.
* Las variables de menor peso (`vehicle_type`, `gearbox`, `fuel_type`) aportan ajuste fino sin dominar la predicción.

### 4. 🛡️ Despliegue Blindado en Streamlit

* **Filtros en cascada:** cada selector restringe los siguientes a las combinaciones que existieron realmente en el histórico, evitando predicciones sobre vehículos imposibles.
* **Sincronización de tipos:** la app convierte las entradas categóricas al mismo `dtype='category'` con el que se entrenó LightGBM, eliminando una fuente clásica de errores silenciosos en inferencia.
* **Ventana temporal de la UI (1990–2015):** los inputs de año se acotan al rango de mayor cobertura del histórico de entrenamiento para que la predicción se mantenga dentro del dominio fiable del modelo.

---

## 🛠️ Tecnologías

* **Lenguaje:** Python 3.x
* **Machine Learning:** LightGBM, XGBoost, CatBoost, scikit-learn, category_encoders, SHAP, joblib
* **Datos:** Pandas, PyArrow (motor Parquet)
* **Despliegue:** Streamlit Cloud

---

## 📁 Estructura del Repositorio

```
.
├── app.py                              # Aplicación Streamlit con filtros en cascada
├── notebooks/
│   └── car_price_regressor.ipynb       # EDA + modelado + SHAP
├── src/
│   ├── preprocesamiento.py             # Limpieza e imputación localizada
│   └── ingenieria_caracteristicas.py   # Pipelines, GridSearch y evaluación
├── datasets/
│   └── df_clean.parquet                # Dataset depurado en formato columnar
├── modelos/
│   └── modelo_lgb_optimizado.joblib    # Modelo LightGBM serializado
├── images/                             # Recursos visuales del README
├── requirements.txt
└── README.md
```

---

## 📫 Contacto

* 💼 [Portafolio](https://davidvaac.github.io/DavidVaAc/#)
* 🌐 [LinkedIn](https://linkedin.com/in/david-fernando-valle-acosta)
* 📋 [Curriculum](https://drive.google.com/file/d/1epmNOV5wLOiH2na0B_kiDaaevGUPrUdF/view?usp=sharing)
* ✉️ [Email](mailto:davidfervalle@gmail.com)
