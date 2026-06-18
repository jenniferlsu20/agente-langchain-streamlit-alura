import streamlit as st
import pandas as pd
import os
import groq
from langchain_groq import ChatGroq
from langchain_core.prompts.prompt import PromptTemplate
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from herramientas import crear_herramientas

# Inicia la aplicación
st.set_page_config(
    page_title="Asistente de Análisis de Datos con IA", layout="centered"
)
st.title("🐦 Asistente de Análisis de Datos con IA")

# Descripción de la herramienta
st.info("""
        Este asistente utiliza un agente, creado con langchain, para ayudarte a explorar, analizar y visualizar datos de forma interactiva.
        Basta con subir un archivo CSV y podrás:
        
        * **Generar reportes automáticos**:
            * **Reporte de información general**: presenta la dimensión del DataFrame, nombres y tipos de columnas, conteo de datos nulos y duplicados, además de sugerir
            * **Reporte de estadísticas descriptivas**: muestra valores como media, mediana, desviación estándar, mínimo y máximo, identifica posibles outliers y sugiere 
            
        * **Hacer preguntas simples sobre los datos**: como "¿Cuál es el promedio de la columna X?", "¿Cuántos registros existen para cada categoría de la columna Y?".
        
        * **Crear gráficos automáticamente** a partir de preguntas en lenguaje natural.
        
        Ideal para analistas, científicos de datos y equipos que buscan agilidad e insights rápidos con apoyo de IA.
        """)

# Upload de CSV
st.markdown("### ⏳ Realiza la carga de tu archivo CSV")
archivo_cargado = st.file_uploader(
    "Selecciona un archivo CSV", type="csv", label_visibility="collapsed"
)

if archivo_cargado:
    df = pd.read_csv(archivo_cargado)
    st.success("✅ Archivo cargado exitosamente")
    st.markdown("## 📄 Primeras filas de tu conjunto de datos")
    st.dataframe(df.head())

    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0,  # llama-3.3-70b-versatile
    )

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

    # Agente
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_es)
    orquestador = AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=25,
        max_execution_time=60.0,
    )

    # Acciones Rápidas
    st.markdown("---")
    st.markdown("## ⚡ Acciones Rápidas")

    # Reporte de informaciones Generales
    if st.button("🖨️ Reporte de Informaciones Generales", key="boton_reporte_general"):
        with st.spinner("Generando Reporte ⏳"):
            respuesta = orquestador.invoke(
                {"input": "Quiero un reporte con información sobre los datos"}
            )
            st.session_state["reporte_general"] = respuesta["output"]

    # Exhibe el reporte con botón de descarga
    if "reporte_general" in st.session_state:
        with st.expander("Resultado: Reporte de Informaciones Generales"):
            st.markdown(st.session_state["reporte_general"])

            st.download_button(
                label="📥📄 Descargar Reporte",
                data=st.session_state["reporte_general"],
                file_name="reporte_informaciones_generales.md",
                mime="text/markdown",
            )

    # Reporte de estadística descriptivas
    if st.button(
        "📄 Reporte de estadísticas descriptivas", key="boton_reporte_estadisticas"
    ):
        with st.spinner("🖨️ Generando Reporte"):
            respuesta = orquestador.invoke(
                {"input": "Quiero un Reporte de estadísticas descriptivas"}
            )
            st.session_state["reporte_estadisticas"] = respuesta["output"]

    # Exhibe el reporte almacenado con opción de descarga
    if "reporte_estadisticas" in st.session_state:
        with st.expander("Resultado: Reporte de Estadísticas Descriptivas"):
            st.markdown(st.session_state["reporte_estadisticas"])

            st.download_button(
                label="⏳ Descargar Reporte",
                data=st.session_state["reporte_estadisticas"],
                file_name="reporte_estadisticas_descriptivas.md",
                mime="text/markdown",
            )

    # Pregunta sobre los datos
    st.markdown("---")
    st.markdown("## 🔍 Preguntas sobre los datos")
    pregunta_sobre_datos = st.text_input(
        "Realiza una pregunta sobre los datos (Ej. 'Cuál es el promedio de tiempo de entrega?')"
    )
    if st.button("Responder pregunta", key="responder_pregunta_dato"):
        with st.spinner("Analizando los Datos..."):
            respuesta = orquestador.invoke({"input": pregunta_sobre_datos})
            st.markdown((respuesta["output"]))

    # Generación de Gráficos
    st.markdown("---")
    st.markdown("## 📈 Crear gráfico con base en una pregunta")

    pregunta_grafico = st.text_input(
        "Qué deseas visualizar? (Ej. 'Genera un gráfico del promedio de tiempo de entrega por clima')"
    )
    if st.button("Genera gráfico", key="generar_grafico"):
        with st.spinner("📥 Generando el gráfico"):
            orquestador.invoke({"input": pregunta_grafico})
