import streamlit as st
import joblib
import pandas as pd

# 1. Cargar el modelo y los datos
model_lgb = joblib.load('modelos/modelo_lgb_optimizado.joblib')
df_clean = pd.read_parquet('datasets/df_clean.parquet')

# Configuración de la página
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.title("Car Price Predictor 🚗")
st.markdown("""Esta aplicación predice el precio de un automóvil basado en sus características. 
Los filtros se actualizan dinámicamente según tus selecciones.""")

st.subheader("Selecciona las características del automóvil")

# --- SECCIÓN 1: Filtros Categóricos en Cascada ---
# Organizamos en 3 columnas para que se vea limpio
col1, col2, col3 = st.columns(3)

# Nivel 1: Marca (Filtro Principal)
brand_options = sorted(df_clean['brand'].dropna().unique())
# format_func permite que se vea como "Bmw" en pantalla, pero guarde "bmw" en la variable
brand = col1.selectbox("Marca", brand_options, format_func=lambda x: str(x).title())

# Nivel 2: Modelo (Se filtra basándose ÚNICAMENTE en la Marca seleccionada)
df_brand = df_clean[df_clean['brand'] == brand]
model_options = sorted(df_brand['model'].dropna().unique())
model = col2.selectbox("Modelo", model_options, format_func=lambda x: str(x).title())

# Nivel 3: Resto de características (Se filtran basándose en la Marca Y el Modelo)
df_model = df_brand[df_brand['model'] == model]

vehicle_type_options = sorted(df_model['vehicle_type'].dropna().unique())
vehicle_type = col3.selectbox("Tipo de vehículo", vehicle_type_options, format_func=lambda x: str(x).title())

# Siguiente fila de columnas
col4, col5, col6 = st.columns(3)

fuel_type_options = sorted(df_model['fuel_type'].dropna().unique())
fuel_type = col4.selectbox("Combustible", fuel_type_options, format_func=lambda x: str(x).title())

gearbox_options = sorted(df_model['gearbox'].dropna().unique())
gearbox = col5.selectbox("Transmisión", gearbox_options, format_func=lambda x: str(x).title())

not_repaired_options = sorted(df_model['not_repaired'].dropna().unique())
not_repaired = col6.selectbox("¿Reparado?", not_repaired_options, format_func=lambda x: str(x).title())


# --- SECCIÓN 2: Filtros Numéricos ---
st.markdown("---") # Una línea divisoria sutil
col7, col8, col9 = st.columns(3)

registration_year = col7.number_input(
    "Año de registro", 
    min_value=1990, 
    max_value=2015, 
    value=2010, 
    help="El modelo está entrenado con datos históricos hasta 2015."
    )
power_options = sorted(df_model['power'].dropna().unique())
power = col8.selectbox("Potencia (CV)", power_options, format_func=lambda x: str(x))
mileage_options = sorted(df_model['mileage'].dropna().unique())
mileage = col9.selectbox("Kilometraje (km)", mileage_options, format_func=lambda x: str(x))


# --- SECCIÓN 3: Predicción ---
st.markdown("<br>", unsafe_allow_html=True) # Espaciado

# Reemplazamos el form_submit_button por un botón normal
if st.button('Predecir Precio', type='tertiary', use_container_width=True):
    
    # Crear un diccionario con las características exactas como las espera el modelo
    car_dict = {
        'vehicle_type': vehicle_type,
        'registration_year': registration_year,
        'gearbox': gearbox,
        'power': power,
        'model': model,
        'mileage': mileage,
        'fuel_type': fuel_type,
        'brand': brand,
        'not_repaired': not_repaired,
    }
    
    # Convertir a DataFrame y ajustar tipos
    car_df = pd.DataFrame([car_dict])
    cat_features = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand', 'not_repaired']
    for col in cat_features:
        car_df[col] = car_df[col].astype('category')

    # Predicción
    predicted_price = model_lgb.predict(car_df)[0]
    
    # Mostrar el resultado de forma destacada
    st.success(f"### 💶 Precio estimado: {predicted_price:,.2f} €")
