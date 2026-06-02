import pandas as pd
import numpy as np

def pascal_case_to_snake_case_cols(df):
    df.columns = df.columns.str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True).str.lower()
    print("✅ Columnas renombradas a snake_case.")
    return df

def imputar_categoricas_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena los valores nulos de las características físicas del auto 
    buscando el valor más común (moda) dentro de su misma marca y modelo.
    """
    df_clean = df.copy()
    
    # Características físicas del vehículo que comparten la misma lógica de imputación
    columnas_a_imputar = ['vehicle_type', 'fuel_type', 'gearbox']
    
    for col in columnas_a_imputar:
        df_clean[col] = df_clean.groupby(['brand', 'model'])[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown')
        )
        
    # Fallback para modelos sin ningún registro válido en esa columna
    for col in columnas_a_imputar:
        df_clean[col] = df_clean[col].fillna('unknown')
    
    print("✅ Valores categóricos imputados por marca y modelo.")

    return df_clean

def corregir_potencia_anomala(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige los CV imposibles (0 o > 400) reemplazándolos con 
    la mediana de potencia de su respectiva marca y modelo.
    """
    df_clean = df.copy()
    
    # Umbrales definidos por criterio de dominio
    limite_inf = 35
    limite_sup = 400
    
    # Valores fuera de rango se marcan como nulos para imputarlos por grupo
    anomalias = (df_clean['power'] <= limite_inf) | (df_clean['power'] > limite_sup)
    df_clean.loc[anomalias, 'power'] = np.nan
    
    # Reemplazamos los nulos con la mediana de la misma marca/modelo
    df_clean['power'] = df_clean.groupby(['brand', 'model'])['power'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fallback global para modelos sin suficientes registros válidos
    df_clean['power'] = df_clean['power'].fillna(df_clean['power'].median())
    print("✅ Potencia anómala corregida.")
    return df_clean

def remover_outliers_precio(df: pd.DataFrame, limite_inferior: int = 100) -> pd.DataFrame:
    """Elimina registros con precios irreales."""
    print(f"✅ Eliminando precios menores a {limite_inferior}")
    return df[df['price'] >= limite_inferior].copy()

def filtrar_años_anomalos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los registros con años imposibles que destruyen la curva de depreciación.
    Límite superior: 2016 (Año de extracción de la base de datos).
    Límite inferior: 1910 (Límite razonable para la existencia de autos clásicos comerciales).
    """
    tamano_original = len(df)
    
    df_clean = df[
        (df['registration_year'] >= 1910) & 
        (df['registration_year'] <= 2016)
    ].copy()
    
    datos_eliminados = tamano_original - len(df_clean)
    print(f"✅ Anomalías de tiempo (viajeros temporales) eliminadas: {datos_eliminados}")
    
    return df_clean

def eliminar_columnas_basura(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas con varianza cero, metadatos y ruido sin poder predictivo."""
    columnas_a_eliminar = [
        'number_of_pictures', 
        'date_crawled', 
        'date_created', 
        'last_seen', 
        'postal_code',
        'registration_month' 
    ]
    print(f"✅ Eliminando columnas irrelevantes para el entrenamiento: {columnas_a_eliminar}")
    return df.drop(columns=columnas_a_eliminar)

def eliminar_duplicados_exactos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas que son idénticas en todas sus columnas.
    Debe ejecutarse DESPUÉS de eliminar las fechas de scraping.
    """
    cantidad_original = len(df)
    df_clean = df.drop_duplicates(keep='first').copy()
    cantidad_final = len(df_clean)
    
    print(f"✅ Duplicados eliminados: {cantidad_original - cantidad_final}")
    
    return df_clean

# Pipeline principal — punto de entrada desde el Notebook
def pipeline_limpieza_base(df: pd.DataFrame) -> pd.DataFrame:
    df = corregir_potencia_anomala(df)
    df = imputar_categoricas_por_modelo(df)
    df['not_repaired'] = df['not_repaired'].fillna('unknown')
    df = df.dropna()
    df = remover_outliers_precio(df)
    df = filtrar_años_anomalos(df)
    df = eliminar_columnas_basura(df)
    df = eliminar_duplicados_exactos(df)
    return df