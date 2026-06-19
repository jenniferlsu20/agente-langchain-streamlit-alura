# 🤖 Agente ReAct para Análisis de Datos con LangChain + Streamlit

Proyecto desarrollado durante el Módulo 3 — **Inteligencia de Datos y RAG Avanzado**  
Oracle Next Education (ONE) con Alura Latam · 2026

🔗 **[Ver aplicación en vivo](https://agente-langchain-app-alura.streamlit.app)**

---

## ¿Qué hace este proyecto?

Un agente inteligente construido con el patrón **ReAct** (Reasoning + Acting) que analiza 
datos en tiempo real. El agente razona sobre la pregunta del usuario, decide qué herramienta 
usar y genera una respuesta basada en datos reales — no en suposiciones.

La aplicación tiene interfaz web desplegada en la nube con Streamlit, accesible desde 
cualquier navegador.

---

## 🛠️ Stack tecnológico

| Herramienta | Rol en el proyecto |
|---|---|
| **LangChain** | Orquestación del agente y cadenas de procesamiento |
| **LangGraph** | Gestión del flujo y memoria del agente |
| **Groq** | Proveedor de inferencia ultrarrápida |
| **Ollama** | Modelo de lenguaje ejecutado vía Groq |
| **ReAct** | Patrón de razonamiento: pensar → actuar → observar |
| **Streamlit** | Interfaz web e implementación en la nube |
| **Python** | Lenguaje base del proyecto |

---

## 🏗️ Arquitectura del agente
Usuario (interfaz Streamlit)

↓

Agente ReAct

┌─────────────────┐

│ 1. Razona       │  ← ¿Qué necesita el usuario?

│ 2. Selecciona   │  ← ¿Qué herramienta usar?

│ 3. Actúa        │  ← Ejecuta herramienta

│ 4. Observa      │  ← Evalúa el resultado

│ 5. Responde     │  ← Genera respuesta final

└─────────────────┘

↓

Respuesta al usuario

---

## 📁 Estructura del repositorio

agente-langchain-streamlit-alura/

│

├── app.py              # Aplicación principal con interfaz Streamlit

├── herramientas.py     # Herramientas disponibles para el agente

├── requirements.txt    # Dependencias del proyecto

└── README.md

---

## 🚀 Cómo ejecutar localmente

1. Clona el repositorio
```bash
git clone https://github.com/jenniferlsu20/agente-langchain-streamlit-alura.git
cd agente-langchain-streamlit-alura
```

2. Instala las dependencias
```bash
pip install -r requirements.txt
```

3. Configura tus variables de entorno (API keys)
```bash
# Crea un archivo .env con tus credenciales
GROQ_API_KEY=tu_clave_aqui
```

4. Ejecuta la aplicación
```bash
streamlit run app.py
```

---

## 📚 Qué aprendí construyendo esto

- Cómo implementar el patrón ReAct para que un agente razone antes de actuar
- Gestión de flujos y memoria con LangGraph
- Construcción de herramientas personalizadas para el agente con LangChain
- Despliegue de aplicaciones de IA en la nube con Streamlit Community Cloud

---

## ⚠️ Estado del proyecto

Proyecto de aprendizaje desarrollado en el programa ONE Oracle + Alura.  
Compartido como evidencia de proceso formativo, no como producto terminado.

---

## 👩‍💻 Sobre mí

Soy Jennifer Silva, profesional en transición hacia tech desde finanzas y administración.  
Actualmente en la Fase 2 de Oracle Next Education (ONE) con Alura Latam.

🔗 [LinkedIn](https://www.linkedin.com/in/jennifer-silva-ustariz-3aa1462a) · 
🌐 [Ver app en vivo](https://agente-langchain-app-alura.streamlit.app)

---

*Oracle Next Education (ONE) · Alura Latam · 2026*

Para actualizar en GitHub desde VS Code:
Abr
