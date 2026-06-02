import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
import catboost as cb
import time

# --- Construcción del preprocesador ---
def construir_preprocesador(cols_numericas: list, cols_cat_alta: list, cols_cat_baja: list) -> ColumnTransformer:
    """Ensambla el ColumnTransformer según las listas de columnas recibidas."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_numericas),
            ('cat_alta', TargetEncoder(), cols_cat_alta),
            ('cat_baja', OneHotEncoder(drop='first', sparse_output=False), cols_cat_baja)
        ],
        remainder='passthrough'
    )

# --- Entrenamiento con búsqueda de hiperparámetros ---
def optimizar_y_entrenar(X_train: pd.DataFrame, y_train: pd.Series, 
                         preprocesador: ColumnTransformer, 
                         algoritmo, 
                         parametros_grid: dict):
    """
    Arma un Pipeline con el preprocesador y el algoritmo dados,
    y ejecuta GridSearchCV para encontrar los mejores hiperparámetros.
    """
    tubo_maestro = Pipeline([
        ('preprocesamiento', preprocesador),
        ('algoritmo', algoritmo)
    ])
    
    grid_search = GridSearchCV(
        estimator=tubo_maestro, 
        param_grid=parametros_grid, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1 
    )
    
    print(f"Iniciando entrenamiento cruzado seguro para {algoritmo.__class__.__name__}...")
    inicio_tiempo = time.time()
    grid_search.fit(X_train, y_train)
    fin_tiempo = time.time()
    
    # Reporte de métricas
    print(f"✅ Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"✅ Mejor RMSE (Validación Cruzada): {-grid_search.best_score_:.2f}")
    print(f"✅ R2 en Train: {grid_search.best_estimator_.score(X_train, y_train):.2f}")
    print(f"⏱️ Tiempo de entrenamiento: {fin_tiempo - inicio_tiempo:.2f} segundos\n")
    
    return [grid_search.best_estimator_, -grid_search.best_score_, grid_search.best_estimator_.score(X_train, y_train), fin_tiempo - inicio_tiempo]

def optimizar_catboost_nativo(X_train: pd.DataFrame, y_train: pd.Series, parametros_grid: dict):
    # Columnas que CatBoost procesará de forma nativa (deben ser dtype string/object)
    columnas_categoricas = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand', 'not_repaired']

    class CatBoost(BaseEstimator, RegressorMixin):

        def __init__(self, iterations=100, learning_rate=0.1, depth=6, random_state=42, cat_features=None, thread_count=-1, verbose=False):
                self.iterations = iterations
                self.learning_rate = learning_rate
                self.depth = depth
                self.random_state = random_state
                self.cat_features = cat_features
                self.thread_count = thread_count
                self.verbose = verbose
                self.model_ = None # Aquí guardaremos el modelo entrenado
        
         # 2. El método fit recibe X e y
        def fit(self, X, y, **kwargs):

            # 2. Extraemos cat_features de los kwargs (si viene desde GridSearchCV.fit)
            # Si no viene, usamos el self.cat_features del constructor            
            cats = kwargs.get('cat_features', self.cat_features)

            # Instanciamos el modelo nativo pasando los parámetros de 'self'
            self.model_ = cb.CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_state=self.random_state,
                verbose=self.verbose # Para que no ensucie la consola durante el GridSearch
            )
        
            # Entrenamos el modelo
            self.model_.fit(X, y, cat_features=cats)
            
            # Scikit-Learn exige que el método fit siempre retorne 'self'
            return self


        def predict(self, X):
            return self.model_.predict(X)
    
    # cat_features se omite en el constructor; pasarlo ahí rompe la clonación interna de GridSearchCV
    algoritmo_cat = CatBoost(
        random_state=42,
        thread_count=-1,
        verbose=False
    )
    
    # Sin Pipeline — CatBoost gestiona su propio preprocesamiento
    grid_search = GridSearchCV(
        estimator=algoritmo_cat, 
        param_grid=parametros_grid, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=1 # n_jobs=1: CatBoost ya paraleliza con thread_count=-1
    )
    
    print("Iniciando entrenamiento cruzado seguro para CatBoost...")
    inicio_tiempo = time.time()
    
    # Pasamos cat_features directo en fit() para sortear la restricción de clonación de sklearn
    grid_search.fit(
        X_train, 
        y_train, 
        cat_features=columnas_categoricas
    )
    fin_tiempo = time.time()
    
    print(f"✅ Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"✅ Mejor RMSE: {-grid_search.best_score_:.2f}")
    print(f"✅ R2 en Train: {grid_search.best_estimator_.score(X_train, y_train):.2f}")
    print(f"⏱️ Tiempo de entrenamiento: {fin_tiempo - inicio_tiempo:.2f} segundos\n")
    
    return [grid_search.best_estimator_, -grid_search.best_score_, grid_search.best_estimator_.score(X_train, y_train), fin_tiempo - inicio_tiempo]

def optimizar_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, parametros_grid: dict):
    """
    Entrena un modelo LightGBM aprovechando su soporte nativo para categorías,
    sin necesidad de TargetEncoder ni StandardScaler.
    Las columnas categóricas deben tener dtype 'category' en X_train antes de llamar esta función.
    """
    algoritmo_lgb = lgb.LGBMRegressor(
        random_state=42, 
        n_jobs=-1,
        #categorical_feature='auto'  # Lee dtypes de Pandas si se activa
        verbose=-1
    )
    
    grid_search = GridSearchCV(
        estimator=algoritmo_lgb, 
        param_grid=parametros_grid, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=1 # n_jobs=1: LightGBM ya paraleliza internamente; apilarlos satura el CPU
    )
    
    print("Iniciando entrenamiento cruzado seguro para LightGBM...")
    inicio_tiempo = time.time()
    grid_search.fit(X_train, y_train)
    fin_tiempo = time.time()
    
    # Reporte de métricas
    print(f"✅ Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"✅ Mejor RMSE: {-grid_search.best_score_:.2f}")
    print(f"✅ R2 en Train: {grid_search.best_estimator_.score(X_train, y_train):.2f}")
    print(f"⏱️ Tiempo de entrenamiento: {fin_tiempo - inicio_tiempo:.2f} segundos\n")
    
    return [grid_search.best_estimator_, -grid_search.best_score_, grid_search.best_estimator_.score(X_train, y_train), fin_tiempo - inicio_tiempo]

def calcular_intervalo_bootstrap(y_true, y_pred, n_muestras=1000):
    """
    Aplica bootstrapping para calcular el intervalo de confianza del 95% del RMSE.
    """
    rmse_scores = []
    n_size = len(y_true)
    
    # Aseguramos que sean arreglos de numpy para indexar fácilmente
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Usamos una semilla fija para reproducibilidad
    estado_aleatorio = np.random.RandomState(42)
    
    for i in range(n_muestras):
        # 1. Crear una muestra aleatoria CON reemplazo (Bootstrap)
        indices = estado_aleatorio.choice(n_size, size=n_size, replace=True)
        muestra_y_true = y_true_np[indices]
        muestra_y_pred = y_pred_np[indices]
        
        # 2. Calcular el RMSE de esta muestra simulada
        rmse = root_mean_squared_error(muestra_y_true, muestra_y_pred)
        rmse_scores.append(rmse)
        
    # 3. Extraer percentiles para el intervalo del 95%
    limite_inferior = float(np.percentile(rmse_scores, 2.5))
    limite_superior = float(np.percentile(rmse_scores, 97.5))
    media_rmse = float(np.mean(rmse_scores))
    
    return media_rmse, limite_inferior, limite_superior

def evaluar_modelos(X_train, y_train, X_test, y_test, models):
    test_results = []
    columnas_categoricas = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand', 'not_repaired']
    for nombre, resultado in models.items():
        if nombre in ['LGBMRegressor']:        
            X_test_cat = X_test.copy()
            X_train_cat = X_train.copy()
            for col in columnas_categoricas:
                X_test_cat[col] = X_test_cat[col].astype('category')
                X_train_cat[col] = X_train_cat[col].astype('category')
            
            tiempo_inicio = time.time()
            resultado[0].fit(X_train_cat, y_train)
            tiempo_fin = time.time()
            tiempo_entrenamiento = tiempo_fin - tiempo_inicio

            # Las conversiones de dtype ya están aplicadas en X_test_cat
            tiempo_inicio = time.time()
            predictions = resultado[0].predict(X_test_cat)
            tiempo_fin = time.time()
            tiempo_prediccion = tiempo_fin - tiempo_inicio

        elif nombre in ['CatBoostRegressor']:
            tiempo_inicio = time.time()
            resultado[0].fit(X_train, y_train,cat_features=columnas_categoricas)
            tiempo_fin = time.time()
            tiempo_entrenamiento = tiempo_fin - tiempo_inicio
            # El Pipeline interno se encarga del preprocesamiento
            tiempo_inicio = time.time()
            predictions = resultado[0].predict(X_test)
            tiempo_fin = time.time()
            tiempo_prediccion = tiempo_fin - tiempo_inicio

        else:
            tiempo_inicio = time.time()
            resultado[0].fit(X_train, y_train)
            tiempo_fin = time.time()
            tiempo_entrenamiento = tiempo_fin - tiempo_inicio
            # Para los demás modelos, el preprocesador espera las columnas originales, así que usamos X_test sin modificaciones
            tiempo_inicio = time.time()
            predictions = resultado[0].predict(X_test)
            tiempo_fin = time.time()
            tiempo_prediccion = tiempo_fin - tiempo_inicio


        media_rmse, limite_inferior, limite_superior = calcular_intervalo_bootstrap(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        test_results.append({
            'modelo': nombre,
            'rmse_2_5': limite_inferior,
            'rmse': media_rmse,
            'rmse_97_5': limite_superior,
            'r2': r2,
            'train_time': tiempo_entrenamiento,
            'predict_time': tiempo_prediccion
        })
    return pd.DataFrame(test_results)