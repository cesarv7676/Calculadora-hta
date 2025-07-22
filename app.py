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
st.write(
    "Esta herramienta utiliza un modelo de Regresión Logística para estimar la probabilidad de que un paciente "
    "logre controlar su presión arterial (<140/90 mmHg) a un año de seguimiento. "
    "Por favor, ingrese los datos del paciente en el panel de la izquierda."
)

# --- PANEL LATERAL PARA LA ENTRADA DE DATOS ---
st.sidebar.header("Datos del Paciente")

# Lista de características en el orden correcto
FINAL_MODEL_FEATURES = [
    'Controles_post',
    'PASpre',
    'EST_Nutricional',
    'Riesgo ACV',
    'Conteo_NUTRi_post',
    'diff_peso',
    'Conteo_Tabact_post',
    'Antigüedad_HTA',
    'SEXOrev',
    'DIETA'
]

def user_input_features():
    """Crea los widgets en el panel lateral para la entrada de datos del usuario."""
    
    # Mapeos para variables categóricas
    map_est_nutricional = {'Bajo Peso': 1, 'Normal': 2, 'Sobrepeso': 3, 'Obesidad': 4}
    map_riesgo_acv = {'Bajo': 1, 'Moderado': 2, 'Alto': 3}
    map_sexo = {'Femenino': 0, 'Masculino': 1}
    map_dieta = {'No sigue indicaciones': 0, 'Sí sigue indicaciones': 1}
    map_antiguedad = {'0 años': 0, '1 año': 1, '2 años': 2, '3 años': 3, '4 años': 4, '5 o más años': 5}

    # --- WIDGETS CON LOS CAMBIOS SOLICITADOS ---

    # 1. "Número de Controles en el último año"
    controles_post = st.sidebar.number_input(
        'Número de Controles en el último año', 
        min_value=0, 
        max_value=5,  # CAMBIO: max=5
        value=3, 
        step=1
    )

    # (PASpre sin cambios)
    paspre = st.sidebar.slider('Presión Arterial Sistólica Inicial (PASpre)', min_value=90, max_value=220, value=145)
    
    # (Estado Nutricional y Riesgo ACV sin cambios)
    est_nutricional_label = st.sidebar.selectbox('Estado Nutricional', options=list(map_est_nutricional.keys()), index=2)
    est_nutricional = map_est_nutricional[est_nutricional_label]

    riesgo_acv_label = st.sidebar.selectbox('Riesgo Cardiovascular (ACV)', options=list(map_riesgo_acv.keys()), index=1)
    riesgo_acv = map_riesgo_acv[riesgo_acv_label]

    # 2. "Número de controles con Nutricionista"
    conteo_nutri_post = st.sidebar.number_input(
        'Número de Controles con Nutricionista', 
        min_value=0, 
        max_value=3,  # CAMBIO: max=3
        value=1, 
        step=1
    )

    # (diff_peso sin cambios)
    diff_peso = st.sidebar.slider('Diferencia de Peso en el último año (kg)', min_value=-20.0, max_value=20.0, value=-2.0, step=0.5)

    # 3. "Consumo diario de Tabaco"
    conteo_tabact_post = st.sidebar.number_input(
        'Consumo diario de Tabaco (cigarrillos)', # CAMBIO: Nueva etiqueta
        min_value=0, 
        max_value=20, # CAMBIO: max=20
        value=0, 
        step=1
    )

    # 4. "Años desde el diagnóstico de Hipertensión"
    antiguedad_hta_label = st.sidebar.selectbox(
        'Años desde el diagnóstico de Hipertensión (HTA)', # CAMBIO: Tipo de widget
        options=list(map_antiguedad.keys())
    )
    antiguedad_hta = map_antiguedad[antiguedad_hta_label]

    # (Sexo y Dieta sin cambios)
    sexorev_label = st.sidebar.selectbox('Sexo Biológico', options=list(map_sexo.keys()))
    sexorev = map_sexo[sexorev_label]
    
    dieta_label = st.sidebar.selectbox('Adherencia a la Dieta', options=list(map_dieta.keys()))
    dieta = map_dieta[dieta_label]
    
    # Crear el diccionario con los datos
    data = {
        'Controles_post': controles_post,
        'PASpre': paspre,
        'EST_Nutricional': est_nutricional,
        'Riesgo ACV': riesgo_acv,
        'Conteo_NUTRi_post': conteo_nutri_post,
        'diff_peso': diff_peso,
        'Conteo_Tabact_post': conteo_tabact_post,
        'Antigüedad_HTA': antiguedad_hta,
        'SEXOrev': sexorev,
        'DIETA': dieta
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# ... (El resto del código para mostrar el DataFrame y calcular la probabilidad no necesita cambios) ...

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
st.info(
    "**Descargo de responsabilidad:** Esta herramienta es un demostrador basado en un modelo predictivo y no reemplaza el juicio clínico profesional. "
    "Las predicciones deben ser interpretadas en el contexto del cuadro clínico completo del paciente."
)