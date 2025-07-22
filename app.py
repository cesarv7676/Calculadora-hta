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

# --- FUNCIÓN PARA CARGAR EL MODELO (con caché para eficiencia) ---
@st.cache_resource
def load_model():
    """Carga el pipeline de preprocesamiento y modelo desde el archivo .pkl"""
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

# --- TÍTULO Y DESCRIPCIÓN DE LA APLICACIÓN ---
st.title("❤️ Calculadora Predictiva de Control de Presión Arterial")
st.write(
    "Esta herramienta utiliza un modelo de Regresión Logística para estimar la probabilidad de que un paciente "
    "logre controlar su presión arterial (<140/90 mmHg) a un año de seguimiento. "
    "Por favor, ingrese los datos del paciente en el panel de la izquierda."
)

# --- PANEL LATERAL PARA LA ENTRADA DE DATOS ---
st.sidebar.header("Datos del Paciente")

def user_input_features():
    """Crea los widgets en el panel lateral para la entrada de datos del usuario."""
    
    # Mapeos para variables categóricas (hace la interfaz más amigable)
    map_est_nutricional = {'Bajo Peso': 1, 'Normal': 2, 'Sobrepeso': 3, 'Obesidad': 4}
    map_riesgo_acv = {'Bajo': 1, 'Moderado': 2, 'Alto': 3} # Asumiendo estos significados
    map_sexo = {'Femenino': 0, 'Masculino': 1} # Asumiendo 0=Femenino, 1=Masculino
    map_dieta = {'No sigue indicaciones': 0, 'Sí sigue indicaciones': 1}

    # Crear los widgets
    controles_post = st.sidebar.number_input('Número de Controles en el último año', min_value=0, max_value=5, value=5, step=1)
    paspre = st.sidebar.slider('Presión Arterial Sistólica Inicial (PASpre)', min_value=90, max_value=220, value=140)
    
    est_nutricional_label = st.sidebar.selectbox('Estado Nutricional', options=list(map_est_nutricional.keys()), index=2)
    est_nutricional = map_est_nutricional[est_nutricional_label]

    riesgo_acv_label = st.sidebar.selectbox('Riesgo Cardiovascular (ACV)', options=list(map_riesgo_acv.keys()), index=1)
    riesgo_acv = map_riesgo_acv[riesgo_acv_label]

    conteo_NUTRi_post = st.sidebar.number_input('Número de Controles con Nutricionista', min_value=0, max_value=3, value=2, step=1)
    diff_peso = st.sidebar.slider('Diferencia de Peso en el último año (kg)', min_value=-20.0, max_value=20.0, value=-2.0, step=0.5)
    conteo_tabact_post = st.sidebar.number_input('Consumo de Tabaco', min_value=0, max_value=20, value=0, step=1)
    Antiguedad_hta = st.sidebar.slider('Años desde el diagnóstico de Hipertensión (HTA)', min_value=0, max_value=5, value=1)

    sexorev_label = st.sidebar.selectbox('Sexo Biológico', options=list(map_sexo.keys()))
    sexorev = map_sexo[sexorev_label]
    
    dieta_label = st.sidebar.selectbox('Adherencia a la Dieta', options=list(map_dieta.keys()))
    dieta = map_dieta[dieta_label]
    
    # Crear un diccionario con los datos
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
    
    # Convertir el diccionario a un DataFrame de pandas
    features = pd.DataFrame(data, index=[0])
    return features

# Obtener las características del usuario
input_df = user_input_features()

# --- MOSTRAR LOS DATOS INGRESADOS ---
st.subheader('Características Ingresadas por el Usuario:')
st.dataframe(input_df, use_container_width=True)


# --- BOTÓN DE PREDICCIÓN Y RESULTADO ---
if st.button('Calcular Probabilidad', key='predict_button'):
    if pipeline is not None:
        try:
            # Realizar la predicción de probabilidad
            prediction_proba = pipeline.predict_proba(input_df)
            
            # Obtener la probabilidad de la clase 1 (Control HTA = 1)
            prob_control = prediction_proba[0][1]
            
            st.subheader('Resultado de la Predicción')
            
            # Usar st.metric para un display visual atractivo
            st.metric(
                label="Probabilidad de Control de HTA a 1 año",
                value=f"{prob_control:.1%}", # Formatear como porcentaje con 1 decimal
                delta=None # Se puede usar para mostrar un cambio si hubiera una comparación
            )
            
            # Añadir una barra de progreso como indicador visual
            st.progress(prob_control)
            
            # Dar una interpretación simple basada en la probabilidad
            if prob_control >= 0.7:
                st.success("El paciente tiene una ALTA probabilidad de lograr el control de su presión arterial. ¡Continuar con el buen trabajo!")
            elif prob_control >= 0.4:
                st.warning("El paciente tiene una probabilidad MODERADA de lograr el control. Se podrían considerar intervenciones de refuerzo.")
            else:
                st.error("El paciente tiene una BAJA probabilidad de lograr el control. Se recomienda una revisión del plan de manejo y seguimiento intensificado.")

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
    else:
        st.error("El modelo no está cargado. No se puede realizar la predicción.")

# --- Información Adicional ---
st.markdown("---")
st.info(
    "**Descargo de responsabilidad:** Esta herramienta es un demostrador basado en un modelo predictivo y no reemplaza el juicio clínico profesional. "
    "Las predicciones deben ser interpretadas en el contexto del cuadro clínico completo del paciente."
)