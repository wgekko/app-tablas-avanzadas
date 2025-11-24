# app-tablas-avanzadas
modelos de tablas y analisis con estadisticas
se tiene que crear una carpeta con punto delante que se llame ".streamlit" y de esa carpeta 
un archivo llamado "config.toml"
y dentro este codigo de configuración de tipo de texto, colores y fuentes

[server]
enableStaticServing = true

[[theme.fontFaces]]
family = "Inter"
url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"

[theme]
primaryColor = "#FF8C00"
backgroundColor = "#0D1B2A"
secondaryBackgroundColor = "#1B263B"
textColor = "#FFA500"
linkColor = "#FFA500"
borderColor = "#CCCCCC"
showWidgetBorder = true
baseRadius = "0.5rem"
buttonRadius = "0.5rem"
font = "Inter"
headingFontWeights = [600, 500]
headingFontSizes = ["2.5rem", "1.8rem"]
codeFont = "Courier New"
codeFontSize = "0.75rem"
codeBackgroundColor = "#112B3C"
showSidebarBorder = false
chartCategoricalColors = [
  "#FF8C00",  # Orange oscuro
  "#FFA500",  # Naranja clásico
  "#FFD700",  # Mostaza / dorado
  "#E1C16E",  # Mostaza claro
  "#C8E25D",  # Lima suave
  "#A8D08D",  # Verde pastel
  "#7AC36A",  # Verde hoja
  "#4CAF50",  # Verde medio
  "#40C4FF",  # Celeste vibrante
  "#00B0F0",  # Celeste profesional
  "#3399FF",  # Celeste más oscuro
  "#1E88E5",  # Azul Francia
  "#1976D2",  # Azul fuerte
  "#1565C0",  # Azul oscuro
  "#0D47A1"   # Azul muy profundo
]

chartCategoricalColors1 = [
  "#FF8C00",
  "#FFA500",
  "#FFB347",
  "#FFD580",
  "#FFA07A",
  "#FF7F50",
  "#FF6F00",
  "#CC7000",
  "#FFC107",
  "#FFDD57",
  "#E67E22",
  "#D35400",
  "#F39C12",
  "#E67E22",
  "#F4A261"
]

[theme.sidebar]
backgroundColor = "#1E3A5F"
secondaryBackgroundColor = "#1B263B"
dataframeHeaderBackgroundColor = "#1A2A40"
headingFontSizes = ["1.6rem", "1.4rem", "1.2rem"]
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Acabo de finalizar el desarrollo de una aplicación interactiva construida en Python y Streamlit para el análisis integral del mercado laboral.
La herramienta procesa bases de datos de empleo, desocupación, subocupación e informalidad, y permite explorar los indicadores a través de visualizaciones avanzadas y modelos estadísticos.

El proyecto integra un stack potente: pandas, numpy, polars, scikit-learn, scipy, statsmodels, plotly, matplotlib, great-tables y gt-extras, lo que posibilita un flujo analítico completo y altamente performante (tal vez con modelos no tan complejos como la app que publique anteriormente).
- Características principales del dashboard:
• Reporte analítico por categorías y tendencias históricas
Tablas dinámicas y estilizadas con Great Tables para explorar la evolución de los indicadores laborales.
• Visualizaciones avanzadas
Sunburst, Treemap, series temporales interactivas, heatmaps y gráficos radar para un entendimiento intuitivo del comportamiento de las tasas laborales.
• Módulo estadístico avanzado
Distribuciones y probabilidades
Clustering y correlaciones (K-Means, Isolation Forest, StandardScaler)
Modelos temporales (STL, ARIMA)
Detección de anomalías y patrones
• Dashboard unificado
Un panel final que resume los principales hallazgos mediante gráficos interactivos y métricas clave.
Este proyecto demuestra cómo el ecosistema Python puede transformar datos complejos en información accesible y accionable, ideal para análisis económico, gestión pública o monitoreo del mercado laboral.


link de los codigos que utilice para la página de inicio y salida de la app
animación de clave 
https://codepen.io/jh3y/pen/JjxPKXz
encriptación de clave
https://codepen.io/dspstudio/pen/OPJZMKX

video demo 



https://github.com/user-attachments/assets/0f41b6c8-dc7e-453c-b62f-c54617e6cf07



