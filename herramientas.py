import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192", temperature=0)


# Herramientas
def crear_herramientas(df: pd.DataFrame, llm):
    """
    Crea las 4 herramientas del agente, todas con acceso al DataFrame
    via closure (en lugar de pasarlo como parámetro de la tool).
    """

    # ── Herramienta 1: Información general del DataFrame ──────────────
    @tool("Informaciones DF", return_direct=True)
    def informaciones_df(pregunta: str) -> str:
        """
        Utiliza esta herramienta siempre que el usuario solicite información
        general sobre el DataFrame: número de columnas y filas, nombres de
        columnas y tipos de datos, conteo de nulos y duplicados.
        """
        shape = df.shape
        columns = df.dtypes
        nulos = df.isnull().sum()
        nans_str = df.apply(
            lambda col: col[~col.isna()]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("nan")
            .sum()
        )
        duplicados = df.duplicated().sum()

        plantilla = PromptTemplate(
            template="""
                    Eres un analista de datos encargado de presentar un resumen informativo
                    sobre un DataFrame a partir de una pregunta hecha por el usuario: {pregunta}

                    ================= INFORMACIÓN DEL DATAFRAME =================
                    Dimensiones: {shape}

                    Columnas y tipos de datos:
                    {columns}

                    Valores nulos por columna:
                    {nulos}

                    Cadenas 'nan' (en cualquier capitalización) por columna:
                    {nans_str}

                    Filas duplicadas: {duplicados}
                    ============================================================

                    C                   on base en esta información, redacta un resumen claro y organizado que contenga:
                    1. Un título: ## Reporte de información general sobre el dataset
                    2. La dimensión total del DataFrame
                    3. La descripción de cada columna (nombre, tipo de dato, qué representa)
                    4. Las columnas con datos nulos y su cantidad
                    5. Las columnas con cadenas 'nan' y su cantidad
                    6. La existencia (o no) de datos duplicados
                    7. Un párrafo sobre los análisis posibles con estos datos
                    8. Un párrafo sobre los tratamientos aplicables a los datos
                    """,
            input_variables=[
                "pregunta",
                "shape",
                "columns",
                "nulos",
                "nans_str",
                "duplicados",
            ],
        )

        cadena = plantilla | llm | StrOutputParser()
        return cadena.invoke(
            {
                "pregunta": pregunta,
                "shape": shape,
                "columns": columns,
                "nulos": nulos,
                "nans_str": nans_str,
                "duplicados": duplicados,
            }
        )

    # ── Herramienta 2: Resumen estadístico ─────────────────────────────
    @tool("Resumen Estadistico", return_direct=True)
    def resumen_estadistico(pregunta: str) -> str:
        """
        Utiliza esta herramienta siempre que el usuario solicite un resumen
        estadístico completo (media, desviación estándar, mínimo, máximo, etc.).
        No la uses para una métrica puntual — para eso usa Herramienta Códigos de Python.
        """
        resumen = df.describe(include="number").transpose().to_string()

        plantilla = PromptTemplate(
            template="""
                    Eres un analista de datos encargado de interpretar resultados estadísticos
                    de una base de datos a partir de esta pregunta: {pregunta}

                    ================= ESTADÍSTICAS DESCRIPTIVAS =================
                    {resumen}
                    ============================================================

                    Elabora un resumen explicativo con lenguaje claro, accesible y fluido:
                    1. Un título: ## Informe de estadísticas descriptivas
                    2. Visión general de las estadísticas de las columnas numéricas
                    3. Un párrafo por cada columna, comentando sus valores
                    4. Identificación de posibles valores atípicos (min/max)
                    5. Recomendaciones de próximos pasos en el análisis
                    """,
            input_variables=["pregunta", "resumen"],
        )

        cadena = plantilla | llm | StrOutputParser()
        return cadena.invoke({"pregunta": pregunta, "resumen": resumen})

    # ── Herramienta 3: Generar gráfico ──────────────────────────────────
    @tool("Generar Grafica", return_direct=True)
    def generar_grafica(pregunta: str) -> str:
        """
        Utiliza esta herramienta cuando el usuario solicite un gráfico del
        DataFrame. Ejemplos: 'crea un gráfico de...', 'plotea...', 'visualiza...',
        'muestra la distribución de...', 'representa gráficamente...'.
        """
        columnas_info = "\n".join(
            f"- {col} ({dtype})" for col, dtype in df.dtypes.items()
        )
        muestra_datos = df.head(3).to_dict(orient="records")

        plantilla = PromptTemplate(
            template="""
                    Eres un especialista en visualización de datos. Genera ÚNICAMENTE el código
                    Python para graficar según la solicitud del usuario.

                    ## Solicitud del usuario:
                    "{pregunta}"

                    ## Metadatos del DataFrame:
                    {columnas}

                    ## Muestra de datos (3 primeras filas):
                    {muestra}

                    ## Instrucciones obligatorias:
                    1. Usa matplotlib.pyplot (como plt) y seaborn (como sns)
                    2. Define el tema con sns.set_theme()
                    3. Todas las columnas mencionadas deben existir en el DataFrame `df`
                    4. Elige el tipo de gráfico según el análisis:
                       - Distribución numérica: histplot, kdeplot, boxplot, violinplot
                       - Distribución categórica: countplot
                       - Comparación entre categorías: barplot
                       - Relación entre variables: scatterplot
                       - Series temporales: lineplot (eje X como fechas)
                    5. figsize=(8, 4)
                    6. Título y etiquetas en los ejes
                    7. Título alineado a la izquierda, pad=20, fontsize=14
                    8. plt.xticks(rotation=0)
                    9. sns.despine()
                    10. Finaliza con plt.show()

                    Devuelve SOLO el código Python, instrucciones separadas por ';',
                    sin texto adicional ni explicación.

                    Código Python:
                    """,
            input_variables=["pregunta", "columnas", "muestra"],
        )

        cadena = plantilla | llm | StrOutputParser()
        script_bruto = cadena.invoke(
            {
                "pregunta": pregunta,
                "columnas": columnas_info,
                "muestra": muestra_datos,
            }
        )

        script_limpio = script_bruto.replace("```python", "").replace("```", "").strip()

        exec_globals = {"df": df, "plt": plt, "sns": sns}
        exec_locals = {}

        try:
            exec(script_limpio, exec_globals, exec_locals)
            fig = plt.gcf()
            st.pyplot(fig)
        finally:
            plt.close("all")

        return "Gráfico generado correctamente."

    # ── Herramienta 4: Ejecutar código Python sobre el df ───────────────
    python_repl = PythonAstREPLTool(locals={"df": df})

    herramienta_codigo_python = python_repl
    herramienta_codigo_python.name = "Herramienta Codigos de Python"
    herramienta_codigo_python.description = """
                                            Utiliza esta herramienta cuando el usuario solicite cálculos, consultas o
                                            transformaciones específicas con Python sobre el DataFrame (df). Ejemplos:
                                            '¿Cuál es el promedio de la columna X?', '¿Valores únicos de la columna Y?',
                                            '¿Correlación entre A y B?'. No la uses para resúmenes generales, estadísticos
                                            completos o gráficos — para eso usa las herramientas específicas.
                                            """

    return [
        informaciones_df,
        resumen_estadistico,
        generar_grafica,
        herramienta_codigo_python,
    ]
