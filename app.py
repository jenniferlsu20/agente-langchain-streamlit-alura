import streamlit as st
import pandas as pd
import os
import groq
from langchain_groq import ChatGroq
from langchain_core.prompts.prompt import PromptTemplate
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from herramientas import crear_herramientas

# Cofiguración de la página
st.set_page_config(
    page_title="Asistente de Análisis de Datos con IA",
    page_icon="🦅",
    layout="wide",  # Cambia la aplicación a pantalla completa
    initial_sidebar_state="collapsed"
)

# Título principal de la aplicación
st.title("🦅 Asistente de Análisis de Datos con IA")
st.markdown("---")

# Selección de Bienvenida
col_info, col_carga = st.columns([6, 4], gap="large")

with col_info:
    # Tarjeta visual con bordes para la explicación del asistente
    with st.container(border=True):
        st.markdown("### 📘 ¿Qué puedes hacer con este asistente?")
        st.markdown("""
        Este asistente utiliza un agente, creado con langchain, para ayudarte a explorar, analizar y visualizar 
        datos de forma interactiva. Basta con subir un archivo CSV y podrás:
        
        * **Generar reportes automáticos:**
            * *Reporte de información general:* presenta la dimensión del DataFrame, nombres y tipos de columnas, conteo de datos nulos y duplicados, además de sugerir.
            * *Reporte de estadísticas descriptivas:* muestra valores como media, mediana, desviación estándar, mínimo y máximo, identifica posibles outliers y sugiere.
        * **Hacer preguntas simples sobre los datos:** como "¿Cuál es el promedio de la columna X?", "¿Cuántos registros existen para cada categoría de la columna Y?".
        * **Crear gráficos automáticamente** a partir de preguntas en lenguaje natural.
        
        Ideal para analistas, científicos de datos y equipos que buscan agilidad e insights rápidos con apoyo de IA.
        """)

with col_carga:
    # Tarjeta visual con bordes para el cargador de archivos CSV
    with st.container(border=True):
        st.markdown("### ⏳ Realiza la carga de tu archivo CSV")
        archivo_subido = st.file_uploader(
            "Upload CSV", 
            type=["csv"],
            help="Límite máximo de 200MB por archivo"
        )

# Procesamiento de datos
if archivo_subido is not None:
    try:
        # Carga del DataFrame
        df = pd.read_csv(archivo_subido)
        
        st.markdown("### 📈 Resumen Estructural del Dataset")
        
        # Tarjeta contenedora para las métricas destacadas (st.metric)
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            
            total_filas = df.shape[0]
            total_columnas = df.shape[1]
            duplicados = df.duplicated().sum()
            nulos_totales = df.isnull().sum().sum()
            
            # Componentes métricos nativos
            m1.metric(label="🔢 Total de Filas", value=f"{total_filas:,}")
            m2.metric(label="📋 Total de Columnas", value=total_columnas)
            m3.metric(
                label="⚠️ Filas Duplicadas", 
                value=duplicados,
                delta=f"-{duplicados} repetidas" if duplicados > 0 else "Limpio",
                delta_color="inverse" if duplicados > 0 else "normal"
            )
            m4.metric(label="🔲 Celdas Vacías (Nulos)", value=f"{nulos_totales:,}")

        # Tarjeta para la visualización previa de la tabla de datos
        with st.container(border=True):
            st.markdown("#### 👀 Vista previa de los primeros 5 registros")
            st.dataframe(df.head(5), use_container_width=True)
            
        # Inicialización del Modelo de Lenguaje (Asegura tener GROQ_API_KEY en tus variables de entorno)
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        
        # Herramientas
        tools = crear_herramientas(df, llm)

        # Prompt react
        df_head = df.head().to_markdown()

        prompt_react_es = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            partial_variables={"df_head": df_head},
            template="""
                    Eres un asistente experto en análisis de datos que responde OBLIGATORIAMENTE en castellano.

                    Tienes acceso a un dataframe pandas llamado `df`.
                    Aquí están las primeras filas del DataFrame:

                    {df_head}

                    Debes responder a la pregunta del usuario utilizando las herramientas disponibles.
                    Tienes acceso a las siguientes herramientas:

                    {tools}

                    REGLA CRÍTICA DE FORMATO: 
                    Debes usar ESTRICTAMENTE las palabras 'Thought:', 'Action:', 'Action Input:', 'Observation:' y 'Final Answer:'. NO LAS TRADUZCAS al español. El sistema interno de LangChain las necesita en inglés. Sin embargo, todo el texto explicativo dentro de ellas DEBE estar en español.

                    Usa el siguiente formato exacto:

                    Question: La pregunta que debes responder.
                    Thought: Tu pensamiento en español sobre lo que debes hacer a continuación.
                    Action: El nombre de la herramienta a usar, debe ser exactamente una de [{tool_names}].
                    Action Input: La entrada exacta para la herramienta. Si usas 'herramienta_codigo_python', escribe SOLO el código de Python puro en una línea, sin bloques de código ``` ni comillas.
                    Observation: El resultado de la acción (esto lo provee el sistema de forma automática).
                    ... (este ciclo de Thought/Action/Action Input/Observation se puede repetir si es necesario)
                    Thought: Pensamiento en español indicando que ya conoces la respuesta final.
                    Final Answer: La respuesta final detallada en castellano para el usuario.

                    ¡Comienza!

                    Question: {input}
                    Thought: {agent_scratchpad}
                    """,
        )

        # Configuración del Agente y Orquestador
        agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_es)
        orquestador = AgentExecutor(
            agent=agente,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=25,
            max_execution_time=60.0,
        )
        
        st.markdown("---")
        st.markdown("## 🧠 Interacción con el Agente de Datos")
        
        # Organización de la salida del Agente mediante Pestañas Elegantes
        tab_preguntas, tab_reportes = st.tabs(["🔍 Consultas Libres", "📊 Reportes de un Clic"])
        
        with tab_preguntas:
            with st.container(border=True):
                st.markdown("### Realiza una pregunta sobre los datos")
                pregunta_sobre_datos = st.text_input(
                    "Escribe tu consulta:",
                    placeholder="Ej. Cuál es el promedio de tiempo de entrega?",
                    key="input_pregunta_libre"
                )
                
                if st.button("Responder pregunta", key="responder_pregunta_dato"):
                    if pregunta_sobre_datos:
                        with st.spinner("Analizando los Datos..."):
                            # Contenedor visual dinámico para renderizar los pensamientos en la interfaz
                            contenedor_pasos = st.container()
                            st_callback = StreamlitCallbackHandler(contenedor_pasos)
                            
                            # Invocación del agente pasando el manejador de callbacks de Streamlit
                            respuesta = orquestador.invoke(
                                {"input": pregunta_sobre_datos},
                                {"callbacks": [st_callback]}
                            )
                            st.session_state["ultima_respuesta_libre"] = respuesta["output"]
                    else:
                        st.warning("⚠️ Por favor, ingresa una pregunta en el cuadro de texto.")
                
                # Muestra la respuesta final persistente bajo los pensamientos
                if "ultima_respuesta_libre" in st.session_state:
                    st.markdown("#### 🎯 Respuesta Final:")
                    st.info(st.session_state["ultima_respuesta_libre"])

        with tab_reportes:
            with st.container(border=True):
                st.markdown("### Generación de Reportes Estructurados Predefinidos")
                st.markdown("Haz clic en cualquiera de los botones para que el agente ejecute los análisis automatizados completos.")
                
                c_rep1, c_rep2 = st.columns(2)
                
                with c_rep1:
                    if st.button("📊 Generar Reporte de Información General", use_container_width=True):
                        with st.spinner("Compilando estructura general..."):
                            contenedor_pasos_rep1 = st.container()
                            st_callback_rep1 = StreamlitCallbackHandler(contenedor_pasos_rep1)
                            res_info = orquestador.invoke(
                                {"input": "Presenta el reporte de información general del dataframe, detallando dimensiones y tipos de datos."},
                                {"callbacks": [st_callback_rep1]}
                            )
                            st.session_state["rep_info_res"] = res_info["output"]
                            
                    if "rep_info_res" in st.session_state:
                        st.success(st.session_state["rep_info_res"])
                        
                with c_rep2:
                    if st.button("🔢 Generar Resumen Estadístico Completo", use_container_width=True):
                        with st.spinner("Calculando estadísticas descriptivas..."):
                            contenedor_pasos_rep2 = st.container()
                            st_callback_rep2 = StreamlitCallbackHandler(contenedor_pasos_rep2)
                            res_est = orquestador.invoke(
                                {"input": "Muestra el resumen estadístico descriptivo completo del dataframe e interpreta las variables numéricas."},
                                {"callbacks": [st_callback_rep2]}
                            )
                            st.session_state["rep_est_res"] = res_est["output"]
                            
                    if "rep_est_res" in st.session_state:
                        st.info(st.session_state["rep_est_res"])

    except Exception as e:
        st.error(f"❌ Ocurrió un error inesperado al procesar el archivo CSV: {str(e)}")
