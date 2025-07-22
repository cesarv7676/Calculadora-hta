import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Calculadora de Control de HTA",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FUNCIÓN PARA CARGAR EL MODELO ---
@st.cache_resource
def load_model():
    """Carga el pipeline del modelo desde el archivo .pkl"""
    try:
        pipeline = joblib.load('final_logistic_regression_model.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo del modelo 'final_logistic_regression_model.pkl'. Asegúrate de que esté en la misma carpeta que app.py.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        return None

# Cargar el modelo
pipeline = load_model()

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("❤️ Calculadora Predictiva de Control de Presión Arterial")
st.write(...) # (Sin cambios aquí)

# --- PANEL LATERAL PARA LA ENTRADA DE DATOS ---
st.sidebar.header("Datos del Paciente")

# --- CORRECCIÓN CLAVE 1: Corregir el nombre en la lista ---
FINAL_MODEL_FEATURES = [
    'Controles_post',
    'PASpre',
    'EST_Nutricional',
    'Riesgo ACV',
    'Conteo_NUTRi_post',  # CORREGIDO: 'i' minúscula
    'diff_peso',
    'Conteo_Tabact_post',
    'Antigüedad_HTA',
    'SEXOrev',
    'DIETA'
]

def user_input_features():
    """Crea los widgets en el panel lateral para la entrada de datos del usuario."""
    
    map_est_nutricional = {'Bajo Peso': 1, 'Normal': 2, 'Sobrepeso': 3, 'Obesidad': 4}
    map_riesgo_acv = {'Bajo': 1, 'Moderado': 2, 'Alto': 3}
    map_sexo = {'Femenino': 0, 'Masculino': 1}
    map_dieta = {'No sigue indicaciones': 0, 'Sí sigue indicaciones': 1}

    # ... (código de los widgets sin cambios) ...
    controles_post = st.sidebar.number_input('Número de Controles en el último año', min_value=0, max_value=50, value=5, step=1)
    paspre = st.sidebar.slider('Presión Arterial Sistólica Inicial (PASpre)', min_value=90, max_value=220, value=145)
    est_nutricional_label = st.sidebar.selectbox('Estado Nutricional', options=list(map_est_nutricional.keys()), index=2)
    est_nutricional = map_est_nutricional[est_nutricional_label]
    riesgo_acv_label = st.sidebar.selectbox('Riesgo Cardiovascular (ACV)', options=list(map_riesgo_acv.keys()), index=1)
    riesgo_acv = map_riesgo_acv[riesgo_acv_label]
    conteo_nutri_post = st.sidebar.number_input('Número de Controles con Nutricionista', min_value=0, max_value=20, value=1, step=1)
    diff_peso = st.sidebar.slider('Diferencia de Peso en el último año (kg)', min_value=-20.0, max_value=20.0, value=-2.0, step=0.5)
    conteo_tabact_post = st.sidebar.number_input('Controles de Tabaquismo', min_value=0, max_value=20, value=0, step=1)
    antiguedad_hta = st.sidebar.slider('Años desde el diagnóstico de Hipertensión (HTA)', min_value=0, max_value=50, value=10)
    sexorev_label = st.sidebar.selectbox('Sexo Biológico', options=list(map_sexo.keys()))
    sexorev = map_sexo[sexorev_label]
    dieta_label = st.sidebar.selectbox('Adherencia a la Dieta', options=list(map_dieta.keys()))
    dieta = map_dieta[dieta_label]
    
    # --- CORRECCIÓN CLAVE 2: Corregir el nombre en la clave del diccionario ---
    data = {
        'Controles_post': controles_post,
        'PASpre': paspre,
        'EST_Nutricional': est_nutricional,
        'Riesgo ACV': riesgo_acv,
        'Conteo_NUTRi_post': conteo_nutri_post, # CORREGIDO: 'i' minúscula
        'diff_peso': diff_peso,
        'Conteo_Tabact_post': conteo_tabact_post,
        'Antigüedad_HTA': antiguedad_hta,
        'SEXOrev': sexorev,
        'DIETA': dieta
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# ... (resto del código sin cambios) ...
input_df = user_input_features()
st.subheader('Características Ingresadas por el Usuario:')
st.dataframe(input_df[FINAL_MODEL_FEATURES], use_container_width=True)

if st.button('Calcular Probabilidad', key='predict_button'):
    if pipeline is not None:
        try:
            input_df_ordered = input_df[FINAL_MODEL_FEATURES]
            prediction_proba = pipeline.predict_proba(input_df_ordered)
            prob_control = prediction_proba[0][1]
            st.subheader('Resultado de la Predicción')
            st.metric(label="Probabilidad de Control de HTA a 1 año", value=f"{prob_control:.1%}")
            st.progress(prob_control)
            if prob_control >= 0.7:
                st.success("El paciente tiene una ALTA probabilidad de lograr el control. ¡Continuar con el buen trabajo!")
            elif prob_control >= 0.4:
                st.warning("El paciente tiene una probabilidad MODERADA de lograr el control. Se podrían considerar intervenciones de refuerzo.")
            else:
                st.error("El paciente tiene una BAJA probabilidad de lograr el control. Se recomienda una revisión del plan de manejo.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
    else:
        st.error("El modelo no está cargado. No se puede realizar la predicción.")

st.markdown("---")
st.info("...") # (Sin cambios aquí)