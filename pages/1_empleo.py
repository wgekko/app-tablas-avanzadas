from great_tables import GT, html, md, style, loc, vals
import polars as pl
import streamlit as st
import pandas as pd
import gt_extras as gte
import numpy as np



# --- Configuración de la página de Streamlit ---
st.set_page_config(layout="wide", page_title="Analsis Tasa Empleo", page_icon="icons/trabajadores.png")

hide_sidebar_style = """
    <style>
        /* Oculta la barra lateral completa */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Ajusta el área principal para usar todo el ancho */
        [data-testid="stAppViewContainer"] {
            margin-left: 0px;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)


# ==============================================================================
# CARGA Y TRANSFORMACIÓN DE DATOS (DATA WRANGLING)
# ==============================================================================

# --- Carga de datos del archivo Excel (.xlsx) ---
# Usamos header=3 para que los títulos de columna (fila 4) sean el encabezado.
# --- Carga de datos del archivo Excel (.xlsx) ---
NOMBRE_ARCHIVO_EXCEL = 'datos/tasa de empleo segun caracteristicas socioeconomicas-actualizado190825.xlsx'

try:
    # Carga con recorte estricto: header=3 para la fila 4; nrows=30 para leer hasta la fila 33.
    dfDatos = pd.read_excel(NOMBRE_ARCHIVO_EXCEL, header=3, nrows=30) 
    st.success(f"Archivo '{NOMBRE_ARCHIVO_EXCEL}' cargado exitosamente. (Recorte estricto aplicado)")
except FileNotFoundError:
    st.error(f"Error: Archivo '{NOMBRE_ARCHIVO_EXCEL}' no encontrado. Por favor, verifica el nombre y la ruta del archivo.")
    st.stop()

# 1. CAPTURAR EL ORDEN ORIGINAL
dfDatos['Orden_Datos'] = dfDatos.index

# Identificar las columnas de Categoría/Item (columnas ID) y asegurarse de que el primer nivel no sea nulo
# La limpieza ahora es más simple gracias al recorte de nrows.
dfDatos = dfDatos.dropna(how='any', subset=dfDatos.columns[:1])
    
# Identificar las columnas de Categoría/Item (columnas ID) y las columnas de tiempo.
id_vars = dfDatos.columns[:2].tolist()
id_vars_melt = id_vars + ['Orden_Datos']
columnas_temporales = dfDatos.columns[2:].tolist() 

# 1. Transformar a formato largo (melt)
dfDatosLargo = dfDatos.melt(
    id_vars=id_vars_melt,
    value_vars=columnas_temporales,
    var_name='Trimestre',
    value_name='Valor_Numérico'
)

# 2. Convertir el valor a numérico y obtener el orden de tiempo
dfDatosLargo['Valor_Numérico'] = pd.to_numeric(dfDatosLargo['Valor_Numérico'], errors='coerce').fillna(0)
dfDatosLargo = dfDatosLargo.dropna(subset=['Valor_Numérico'])

# Crear código de trimestre para asegurar el orden temporal
dfDatosLargo['Orden_Trimestre'] = dfDatosLargo['Trimestre'].astype('category').cat.codes

# 3. CÁLCULO: Encontrar el valor anterior (T-1) y el valor del año anterior (YoY)
dfDatosLargo = dfDatosLargo.sort_values(by=id_vars + ['Orden_Trimestre'])

# Valor T-1 (Trimestre anterior, 1 período atrás)
dfDatosLargo['Valor_Anterior'] = dfDatosLargo.groupby(id_vars)['Valor_Numérico'].shift(1).fillna(0)
# Valor YoY (Mismo trimestre del año anterior, 4 períodos atrás)
dfDatosLargo['Valor_YoY'] = dfDatosLargo.groupby(id_vars)['Valor_Numérico'].shift(4).fillna(0)

# Calcular el CAMBIO PORCENTUAL (vs T-1)
dfDatosLargo['Cambio_Porcentual_Reciente'] = np.where(
    dfDatosLargo['Valor_Anterior'] != 0,
    (dfDatosLargo['Valor_Numérico'] - dfDatosLargo['Valor_Anterior']) / dfDatosLargo['Valor_Anterior'],
    0
)

# Calcular el CAMBIO PORCENTUAL VS AÑO ANTERIOR (YoY)
dfDatosLargo['Cambio_Porcentual_YoY'] = np.where(
    dfDatosLargo['Valor_YoY'] != 0,
    (dfDatosLargo['Valor_Numérico'] - dfDatosLargo['Valor_YoY']) / dfDatosLargo['Valor_YoY'],
    0
)

# 4. Preparación del DataFrame Final para la Tabla GT (Tendencia Histórica)
dfDatosTabla = dfDatosLargo.groupby(id_vars + ['Orden_Datos'])["Valor_Numérico"].apply(list).reset_index()

# 5. Obtener los Valores Más Recientes 
trimestre_reciente = dfDatosLargo['Trimestre'].max()
dfValoresRecientes = dfDatosLargo[dfDatosLargo['Trimestre'] == trimestre_reciente]

dfValoresRecientes = dfValoresRecientes.rename(columns={'Valor_Numérico': 'Valor_Reciente', 'Trimestre': 'Trimestre_Reciente'})
dfValoresRecientes = dfValoresRecientes[id_vars + ['Orden_Datos', 'Valor_Reciente', 'Trimestre_Reciente', 
                                                   'Cambio_Porcentual_Reciente', 'Cambio_Porcentual_YoY']]

# 6. Unir la tendencia y los valores recientes
dfDatosTabla = dfDatosTabla.merge(dfValoresRecientes, on=id_vars + ['Orden_Datos'], how='left')

# --- Preparación de Variables para Streamlit ---
nombre_col_categoria = id_vars[0]
nombre_col_item = id_vars[1]
trimestre_reciente_display = dfDatosTabla['Trimestre_Reciente'].max()


# ==============================================================================
# CÁLCULOS PARA MODELOS DE ANÁLISIS RESUMEN POR CATEGORÍA
# ==============================================================================

# 1. Obtener el Promedio Histórico para cada Categoría
dfPromedioHistorico = dfDatosLargo.groupby(nombre_col_categoria)['Valor_Numérico'].mean().reset_index()
dfPromedioHistorico = dfPromedioHistorico.rename(columns={'Valor_Numérico': 'Promedio_Historico'})

# 2. Obtener el Valor Reciente y el Cambio YoY promedio por Categoría
dfValorRecienteCat = dfValoresRecientes.groupby(nombre_col_categoria).agg({
    'Valor_Reciente': 'mean',
    'Cambio_Porcentual_YoY': 'mean' 
}).reset_index()

# 3. Unir los DataFrames y calcular la métrica de análisis
dfAnalisis = dfPromedioHistorico.merge(dfValorRecienteCat, on=nombre_col_categoria, how='left')

# Calcular el Cambio Porcentual vs Promedio Histórico (PH)
dfAnalisis['Cambio_vs_PH'] = np.where(
    dfAnalisis['Promedio_Historico'] != 0,
    (dfAnalisis['Valor_Reciente'] - dfAnalisis['Promedio_Historico']) / dfAnalisis['Promedio_Historico'],
    0
)

# Seleccionar y renombrar para la tabla resumen
dfAnalisis = dfAnalisis[[nombre_col_categoria, 'Valor_Reciente', 'Promedio_Historico', 
                         'Cambio_vs_PH', 'Cambio_Porcentual_YoY']]
dfAnalisis = dfAnalisis.rename(columns={
    'Valor_Reciente': f'Valor Reciente ({trimestre_reciente_display})',
    'Promedio_Historico': 'Promedio Histórico',
    'Cambio_Porcentual_YoY': 'Cambio % (YoY Promedio)'
})

with st.expander("Informe de datos de Empleo- Tablas- avamzadas", expanded=False, icon=":material/work_alert:", width="stretch"):
        # ==============================================================================
        # CREACIÓN DE LA INTERFAZ DE USUARIO CON STREAMLIT
        # ==============================================================================
        st.title(
            "Reporte de Datos por Categoría y Tendencia Histórica :material/breaking_news:",
            help="La tabla mantiene el orden original de la fuente de datos."
        )

        parCategoria = st.multiselect(
            f"Selecciona la/las {nombre_col_categoria}(s) a mostrar",
            dfDatosTabla[nombre_col_categoria].unique().tolist(),
            default=dfDatosTabla[nombre_col_categoria].unique().tolist()
        )

        parOrdenarPor = st.selectbox(
            "Ordenar tabla por:",
            options=["Orden_Datos", "Valor_Reciente"],
            format_func=lambda x: "Orden Original" if x == "Orden_Datos" else "Valor Reciente (Decreciente)", 
            index=0 
        )

        orden_ascendente = parOrdenarPor == "Orden_Datos"

        if parCategoria:
            dfDatosTabla = dfDatosTabla[dfDatosTabla[nombre_col_categoria].isin(parCategoria)] 

        # ==============================================================================
        # GENERACIÓN DE LA TABLA PRINCIPAL (DETALLE)
        # ==============================================================================

        tabTabla, tabDatos = st.tabs([f"Tabla {nombre_col_categoria}", "Datos Fuente"])

        with tabTabla:
            table = (
                GT(
                    pl.from_pandas(dfDatosTabla.sort_values(parOrdenarPor, ascending=orden_ascendente)),
                    rowname_col=nombre_col_item,
                    groupname_col=nombre_col_categoria
                )
                .cols_hide("Orden_Datos")
                .tab_header(
                    title=html("Visualización de Valores por Trimestre y Categoría"),
                    subtitle=html(f"Ordenamiento por Categoría e Ítem. Valor más reciente al **{trimestre_reciente_display}**."),
                )
                .fmt_nanoplot("Valor_Numérico")
                .fmt_number(
                    columns="Valor_Reciente",
                    decimals=0
                )
                # Formateo de los dos cambios porcentuales
                .fmt_percent(
                    columns=["Cambio_Porcentual_Reciente", "Cambio_Porcentual_YoY"],
                    decimals=1 
                )
                .cols_label(
                    **{"Valor_Numérico": "Tendencia Histórica"},
                    **{"Valor_Reciente": "Valor Reciente"},
                    **{"Cambio_Porcentual_Reciente": "Cambio % (vs T-1)"}, 
                    **{"Cambio_Porcentual_YoY": "Cambio % (vs Año Ant.)"},
                    **{"Trimestre_Reciente": "Período Reciente"}
                )
                
                # Estilos Condicionales para T-1
                .tab_style(
                    style=style.text(color="red", weight="bold"),
                    locations=loc.body(columns="Cambio_Porcentual_Reciente", rows=pl.col("Cambio_Porcentual_Reciente") < 0),
                )
                .tab_style(
                    style=style.text(color="green", weight="bold"),
                    locations=loc.body(columns="Cambio_Porcentual_Reciente", rows=pl.col("Cambio_Porcentual_Reciente") > 0),
                )
                # Estilos Condicionales para YoY
                .tab_style(
                    style=style.text(color="red", weight="bold"),
                    locations=loc.body(columns="Cambio_Porcentual_YoY", rows=pl.col("Cambio_Porcentual_YoY") < 0),
                )
                .tab_style(
                    style=style.text(color="green", weight="bold"),
                    locations=loc.body(columns="Cambio_Porcentual_YoY", rows=pl.col("Cambio_Porcentual_YoY") > 0),
                )
                
                # Estilos visuales
                .tab_style(
                    style=[style.text(weight="bold"), style.fill(color="#D9E9CF")],
                    locations=loc.row_groups() 
                )
                .tab_style(
                    style=style.text(weight="bold"),
                    locations=loc.body(columns=nombre_col_item) 
                )
                .tab_source_note(source_note="Fuente: Datos de la planilla Excel.")
                .as_raw_html()
            )
            st.write(table, unsafe_allow_html=True) 
            
        with tabDatos:
            st.subheader("Datos Fuente - Transformados y Agrupados")
            st.dataframe(dfDatosTabla)
        # ==============================================================================
        # GENERACIÓN DE MODELOS DE ANÁLISIS RESUMEN (CATEGORÍA)
        # ==============================================================================
        st.markdown("---")
        st.header("Modelos de Análisis Resumen por Categoría")
        st.caption(f"Comparación del Valor Reciente ({trimestre_reciente_display}) vs Promedio Histórico y Año Anterior (YoY).")

        # Crear la tabla de análisis con GT
        table_analisis = (
            GT(pl.from_pandas(dfAnalisis), rowname_col=nombre_col_categoria)
            
            # Formatear la columna de Valor Reciente y Promedio Histórico (Números con 1 decimal)
            .fmt_number(
                columns=[f'Valor Reciente ({trimestre_reciente_display})', 'Promedio Histórico'],
                decimals=1
            )
            # Formatear las columnas de Cambio Porcentual (con %)
            .fmt_percent(
                columns=['Cambio_vs_PH', 'Cambio % (YoY Promedio)'],
                decimals=1
            )
            # Aplicar estilos condicionales (rojo/verde) al Cambio vs PH
            .tab_style(
                style=style.text(color="red", weight="bold"),
                locations=loc.body(columns='Cambio_vs_PH', rows=pl.col("Cambio_vs_PH") < 0),
            )
            .tab_style(
                style=style.text(color="green", weight="bold"),
                locations=loc.body(columns='Cambio_vs_PH', rows=pl.col("Cambio_vs_PH") > 0),
            )
            # Aplicar estilos condicionales (rojo/verde) al Cambio YoY Promedio
            .tab_style(
                style=style.text(color="red", weight="bold"),
                locations=loc.body(columns='Cambio % (YoY Promedio)', rows=pl.col("Cambio % (YoY Promedio)") < 0),
            )
            .tab_style(
                style=style.text(color="green", weight="bold"),
                locations=loc.body(columns='Cambio % (YoY Promedio)', rows=pl.col("Cambio % (YoY Promedio)") > 0),
            )
            .as_raw_html()
        )

        st.write(table_analisis, unsafe_allow_html=True)

# ===============================================================================================================
# GENERACIÓN DE MODELOS DE ANÁLISIS CON GRAFICOS SE GENERA UN CODIGO CON POTENCIAL USO INDEPENDIENTE DEL ANTERIOR 
# ===============================================================================================================
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from great_tables import GT, html, style, loc
import gt_extras as gte

with st.expander("Informe con Gráficos Sunburst/Treemap, Líneas Dinámicas, Heatmap, Radar por Categorias", expanded=False, icon=":material/work_alert:", width="stretch"):
        
        
    # Ruta al archivo (ajustar si es necesario)
    NOMBRE_ARCHIVO_EXCEL = 'datos/tasa de empleo segun caracteristicas socioeconomicas-actualizado190825.xlsx'

    # --- Carga segura del Excel ---
    if not Path(NOMBRE_ARCHIVO_EXCEL).exists():
        st.error(f"Archivo no encontrado: {NOMBRE_ARCHIVO_EXCEL}")
        st.stop()

    df = pd.read_excel(NOMBRE_ARCHIVO_EXCEL, header=3)
    if df.shape[0] > 0:
        df = df.dropna(how='all')

    df['Orden_Datos'] = range(len(df))

    # ------------------------
    # Detección automática de jerarquía
    # ------------------------
    max_hierarchy_cols = 4
    candidate_cols = df.columns.tolist()[:max_hierarchy_cols]

    id_vars = []
    for c in candidate_cols:
        non_null_ratio = df[c].notna().mean()
        if non_null_ratio > 0.1:
            id_vars.append(c)
        else:
            break
    if len(id_vars) < 2:
        id_vars = df.columns[:2].tolist()

    col_index_start = df.columns.get_loc(id_vars[-1]) + 1
    columnas_temporales = df.columns[col_index_start:].tolist()

    nombre_col_categoria = id_vars[0]
    nombre_col_item = id_vars[1] if len(id_vars) > 1 else id_vars[0]

    # --- Melt a formato largo ---
    df_largo = df.melt(id_vars=id_vars + ['Orden_Datos'], value_vars=columnas_temporales,
                        var_name='Trimestre', value_name='Valor_Num')

    df_largo['Valor_Num'] = pd.to_numeric(df_largo['Valor_Num'], errors='coerce')
    df_largo = df_largo.dropna(subset=['Valor_Num']).reset_index(drop=True)

    def try_parse_time_labels(series):
        parsed = pd.to_datetime(df_largo['Trimestre'], errors='coerce', format='mixed')
        if parsed.notna().any():
            return parsed
        return pd.Categorical(series, categories=pd.unique(series), ordered=True)

    parsed_times = try_parse_time_labels(df_largo['Trimestre'])
    if isinstance(parsed_times, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(parsed_times):
        df_largo['Orden_Trimestre'] = pd.to_datetime(df_largo['Trimestre']).rank(method='dense').astype(int)
    else:
        df_largo['Orden_Trimestre'] = pd.Categorical(df_largo['Trimestre'], categories=pd.unique(df_largo['Trimestre']), ordered=True).codes

    df_largo = df_largo.sort_values(by=id_vars + ['Orden_Trimestre']).reset_index(drop=True)

    df_largo['Valor_Anterior'] = df_largo.groupby(id_vars)['Valor_Num'].shift(1)
    df_largo['Valor_YoY'] = df_largo.groupby(id_vars)['Valor_Num'].shift(4)

    df_largo['Cambio_T1'] = np.where(df_largo['Valor_Anterior'].notna() & (df_largo['Valor_Anterior'] != 0),
                                    (df_largo['Valor_Num'] - df_largo['Valor_Anterior']) / df_largo['Valor_Anterior'],
                                    np.nan)

    df_largo['Cambio_YoY'] = np.where(df_largo['Valor_YoY'].notna() & (df_largo['Valor_YoY'] != 0),
                                    (df_largo['Valor_Num'] - df_largo['Valor_YoY']) / df_largo['Valor_YoY'],
                                    np.nan)

    # --- Nuevas métricas por categoría (Capa 1) ---
    pendientes = []
    volatilidad = []
    percentiles = []
    for name, g in df_largo.groupby(nombre_col_categoria):
        if g['Orden_Trimestre'].nunique() >= 2:
            X = g['Orden_Trimestre'].values.reshape(-1, 1)
            y = g['Valor_Num'].values
            lr = LinearRegression()
            lr.fit(X, y)
            pendientes.append((name, lr.coef_[0]))
            volatilidad.append((name, g['Valor_Num'].std()))
            p25, p50, p75 = np.percentile(y, [25, 50, 75])
            percentiles.append((name, p25, p50, p75))
        else:
            pendientes.append((name, np.nan))
            volatilidad.append((name, np.nan))
            percentiles.append((name, np.nan, np.nan, np.nan))

    df_pend = pd.DataFrame(pendientes, columns=[nombre_col_categoria, 'Pendiente_Historica'])
    df_vol = pd.DataFrame(volatilidad, columns=[nombre_col_categoria, 'Volatilidad'])
    df_pct = pd.DataFrame(percentiles, columns=[nombre_col_categoria, 'P25', 'P50', 'P75'])

    ultimo_trimestre = df_largo['Trimestre'].unique()[-1]
    df_reciente = df_largo[df_largo['Trimestre'] == ultimo_trimestre].copy()
    df_reciente = df_reciente.rename(columns={'Valor_Num': 'Valor_Reciente'})

    df_prom = df_largo.groupby(nombre_col_categoria)['Valor_Num'].mean().reset_index().rename(columns={'Valor_Num': 'Promedio_Historico'})

    df_analisis = df_prom.merge(df_reciente.groupby(nombre_col_categoria)['Valor_Reciente'].mean().reset_index(), on=nombre_col_categoria, how='left')
    df_analisis = df_analisis.merge(df_pend, on=nombre_col_categoria, how='left')
    df_analisis = df_analisis.merge(df_vol, on=nombre_col_categoria, how='left')
    df_analisis = df_analisis.merge(df_pct, on=nombre_col_categoria, how='left')
    df_analisis['Cambio_vs_PH'] = np.where(df_analisis['Promedio_Historico'] != 0,
                                        (df_analisis['Valor_Reciente'] - df_analisis['Promedio_Historico']) / df_analisis['Promedio_Historico'],
                                        np.nan)
    df_analisis['Index_100'] = 100 * df_analisis['Valor_Reciente'] / df_analisis['Promedio_Historico']

    df_analisis = df_analisis.sort_values('Valor_Reciente', ascending=False).reset_index(drop=True)
    df_analisis['Rank_Valor_Reciente'] = df_analisis['Valor_Reciente'].rank(method='min', ascending=False).astype(int)

    df_tendencia = df_largo.groupby([nombre_col_categoria, nombre_col_item, 'Orden_Datos'])['Valor_Num'].apply(list).reset_index()
    ult = df_reciente[[nombre_col_categoria, nombre_col_item, 'Valor_Reciente', 'Cambio_T1', 'Cambio_YoY', 'Trimestre']]
    ult = ult.rename(columns={'Trimestre':'Trimestre_Reciente'})
    if {nombre_col_categoria, nombre_col_item}.issubset(set(df_tendencia.columns)):
        df_tendencia = df_tendencia.merge(ult, on=[nombre_col_categoria, nombre_col_item], how='left')

    # ---- UI: filtros ----
    st.header('Reporte con Graficos, Líneas dinámicas, Heatmap temporal, Radar x categoría')
    col1, col2,col3 = st.columns([1,3,1])
    with col2:
        st.header('Filtros')
        categorias = df_largo[nombre_col_categoria].dropna().unique().tolist()
        sel_cats = st.multiselect('Selecciona categorías', categorias, default=categorias)
        mostrar_sunburst = st.checkbox('Mostrar Sunburst/Treemap', value=True)
        mostrar_lineas = st.checkbox('Mostrar Líneas dinámicas', value=True)
        mostrar_heatmap = st.checkbox('Mostrar Heatmap temporal', value=False)
        mostrar_radar = st.checkbox('Mostrar Radar por categoría', value=False)


        

    # Aplicar filtros a df_largo
    df_filtrado = df_largo[df_largo[nombre_col_categoria].isin(sel_cats)]

    # --- Visualizaciones ---
    st.markdown('---')

    # 1) Sunburst / Treemap
    if mostrar_sunburst:
        st.subheader('Sunburst / Treemap - Valor Reciente por Jerarquía')
        try:
            fig_sun = px.sunburst(df_reciente, path=[nombre_col_categoria, nombre_col_item], values='Valor_Reciente',
                                color='Cambio_YoY', hover_data=['Valor_Reciente'])
            st.plotly_chart(fig_sun, use_container_width=True)
        except Exception as e:
            st.error('Error al generar Sunburst: ' + str(e))
        st.caption('Sunburst coloreado por Cambio Año a Año. Treemap abajo.')
        try:
            fig_tree = px.treemap(df_reciente, path=[nombre_col_categoria, nombre_col_item], values='Valor_Reciente',
                                color='Cambio_YoY', hover_data=['Valor_Reciente'])
            st.plotly_chart(fig_tree, use_container_width=True)
        except Exception as e:
            st.error('Error al generar Treemap: ' + str(e))

    # 2) Líneas dinámicas
    if mostrar_lineas:
        st.subheader('Líneas dinámicas por Trimestre')
        fig_line = px.line(df_filtrado, x='Trimestre', y='Valor_Num', color=nombre_col_categoria, line_group=nombre_col_item,
                        markers=True, hover_data=[nombre_col_item])
        fig_line.update_layout(legend_title=nombre_col_categoria)
        st.plotly_chart(fig_line, use_container_width=True)

    # 3) Slope graph
    st.subheader('Slope Graph: Último Trimestre vs Promedio Histórico')
    try:
        df_slope = df_analisis[[nombre_col_categoria, 'Valor_Reciente', 'Promedio_Historico']].dropna()
        df_slope_m = pd.melt(df_slope, id_vars=[nombre_col_categoria], value_vars=['Promedio_Historico', 'Valor_Reciente'],
                            var_name='Periodo', value_name='Valor')
        fig_slope = px.line(df_slope_m, x='Periodo', y='Valor', color=nombre_col_categoria, markers=True)
        for i, row in df_slope.iterrows():
            fig_slope.add_trace(go.Scatter(x=['Promedio_Historico'], y=[row['Promedio_Historico']], mode='text',
                                        text=[row[nombre_col_categoria]], showlegend=False))
        st.plotly_chart(fig_slope, use_container_width=True)
    except Exception as e:
        st.error('Error Slope Graph: ' + str(e))

    # 4) Heatmap temporal
    if mostrar_heatmap:
        st.subheader('Heatmap temporal por categoría y trimestre')
        try:
            pivot = df_filtrado.groupby([nombre_col_categoria, 'Trimestre'])['Valor_Num'].mean().reset_index()
            pivot_p = pivot.pivot(index=nombre_col_categoria, columns='Trimestre', values='Valor_Num')
            fig_h = go.Figure(data=go.Heatmap(z=pivot_p.values, x=pivot_p.columns.tolist(), y=pivot_p.index.tolist(), hoverongaps=False))
            fig_h.update_layout(height=600)
            st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.error('Error Heatmap: ' + str(e))

    # 5) Radar chart
    if mostrar_radar:
        st.subheader('Radar por categoría (normalizado)')
        try:
            topn = st.slider('Top N categorías para radar', 1, min(10, len(df_analisis)), 5)
            radar_df = df_analisis.sort_values('Valor_Reciente', ascending=False).head(topn)
            metrics = ['Valor_Reciente', 'Promedio_Historico', 'Cambio_vs_PH', 'Volatilidad']
            norm = (radar_df[metrics] - radar_df[metrics].min()) / (radar_df[metrics].max() - radar_df[metrics].min())
            categories_axis = ['Valor reciente', 'Promedio histórico', 'Cambio vs PH', 'Volatilidad']
            fig_r = go.Figure()
            for i, r in norm.iterrows():
                fig_r.add_trace(go.Scatterpolar(r=r.values.tolist(), theta=categories_axis, fill='toself', name=radar_df.loc[i, nombre_col_categoria]))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(fig_r, use_container_width=True)
        except Exception as e:
            st.error('Error Radar: ' + str(e))

    # 6) Sparklines
    st.subheader('Sparklines por Ítem (últimas N períodos)')
    try:
        last_n = st.slider('Últimos períodos para sparkline', 3, 20, 6)
        sample_items = df_filtrado[nombre_col_item].dropna().unique().tolist()[:50]
        cols = st.columns(3)
        cnt = 0
        for item in sample_items[:12]:
            sub = df_largo[df_largo[nombre_col_item] == item].sort_values('Orden_Trimestre')
            y = sub['Valor_Num'].tail(last_n).values
            x = sub['Trimestre'].tail(last_n).values
            fig_s = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', showlegend=False))
            fig_s.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10))
            cols[cnt % 3].plotly_chart(fig_s, use_container_width=True)
            cnt += 1
    except Exception as e:
        st.error('Error Sparklines: ' + str(e))

    with st.expander("Tabla con tendencia histórica y analisis por Categoría", expanded=False, icon=":material/table_sign:", width="stretch"):
        # --- Tabla GT ---
        st.markdown('---')
        st.header('Tabla Detalle Mejorada')
        try:
            table_html = (
                GT(
                    pl.from_pandas(df_tendencia.sort_values('Orden_Datos')),
                    rowname_col=nombre_col_item,
                    groupname_col=nombre_col_categoria
                )
                .cols_hide('Orden_Datos')
                .fmt_number(columns='Valor_Reciente', decimals=1)
                .fmt_percent(columns=['Cambio_T1', 'Cambio_YoY'], decimals=1)
                .cols_label(**{'Valor_Num': 'Tendencia Histórica', 'Valor_Reciente': 'Valor Reciente'})
                .tab_style(style=style.text(weight='bold'), locations=loc.row_groups())
                .tab_source_note(source_note='Fuente: Excel')
                .as_raw_html()
            )
            st.write(table_html, unsafe_allow_html=True)
        except Exception as e:
            st.error('Error GT table: ' + str(e))

        # --- Tabla Analisis resumido ---
        st.header('Análisis Resumido por Categoría')
        try:
            df_anal_display = df_analisis.copy()
            def trend_badge(v):
                if pd.isna(v):
                    return ''
                if v > 0:
                    return '↑'
                elif v < 0:
                    return '↓'
                else:
                    return '→'

            df_anal_display['Tendencia_Badge'] = df_anal_display['Pendiente_Historica'].apply(trend_badge)
            df_anal_display['Index_100'] = df_anal_display['Index_100'].round(1)

            table_anal = (
                GT(pl.from_pandas(df_anal_display), rowname_col=nombre_col_categoria)
                .fmt_number(columns=['Valor_Reciente', 'Promedio_Historico', 'Index_100'], decimals=1)
                .fmt_percent(columns=['Cambio_vs_PH'], decimals=1)
                .cols_label(**{'Valor_Reciente': f'Valor Reciente ({ultimo_trimestre})', 'Cambio_vs_PH': 'Cambio vs PH'})
                .tab_style(style=style.text(weight='bold'), locations=loc.row_groups())
                .as_raw_html()
            )
            st.write(table_anal, unsafe_allow_html=True)
        except Exception as e:
            st.error('Error tabla análisis: ' + str(e))

    st.markdown('---')

# ==============================================================================================================================
# GENERACIÓN DE MODELOS DE ANÁLISIS CON ESTADISTICAS AVANZADAS SE GENERA UN CODIGO CON POTENCIAL USO INDEPENDIENTE DEL ANTERIOR 
# ===============================================================================================================================

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# visual
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# stats
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy

# time series
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA


with st.expander("Modelo Estadístico Avanzado", expanded=False, icon=":material/work_alert:", width="stretch"):

    # ----------------------------
    # Config & carga archivo
    # ----------------------------
    FILE_PATH = "datos/tasa de empleo segun caracteristicas socioeconomicas-actualizado190825.xlsx"

    st.header("Dashboard Estadístico Avanzado — Tasa de Empleo")
    st.caption("Módulos: Distribuciones, Probabilidades, Clustering, Modelos temporales, Visualizaciones interactivas")

    # Verificar existencia
    if not Path(FILE_PATH).exists():
        st.error(f"Archivo no encontrado en: {FILE_PATH}")
        st.stop()

    @st.cache_data
    def load_raw(path):
        df = pd.read_excel(path, header=3)
        df = df.dropna(how='all')
        # Guardar copia original
        df['Orden_Datos'] = range(len(df))
        return df

    raw = load_raw(FILE_PATH)

    # Detección sencilla de columnas jerárquicas (como en tu código base)
    max_hierarchy_cols = 4
    candidate_cols = raw.columns.tolist()[:max_hierarchy_cols]

    id_vars = []
    for c in candidate_cols:
        non_null_ratio = raw[c].notna().mean()
        if non_null_ratio > 0.1:
            id_vars.append(c)
        else:
            break
    if len(id_vars) < 2:
        id_vars = raw.columns[:2].tolist()

    col_index_start = raw.columns.get_loc(id_vars[-1]) + 1
    columnas_temporales = raw.columns[col_index_start:].tolist()

    nombre_col_categoria = id_vars[0]
    nombre_col_item = id_vars[1] if len(id_vars) > 1 else id_vars[0]

    # Convertir a formato largo
    @st.cache_data
    def to_long(df):
        df_largo = df.melt(id_vars=id_vars + ['Orden_Datos'], value_vars=columnas_temporales,
                            var_name='Trimestre', value_name='Valor_Num')
        df_largo['Valor_Num'] = pd.to_numeric(df_largo['Valor_Num'], errors='coerce')
        df_largo = df_largo.dropna(subset=['Valor_Num']).reset_index(drop=True)
        # crear orden temporal
        try:
            #parsed = pd.to_datetime(df_largo['Trimestre'], errors='coerce')
            parsed = pd.to_datetime(df_largo['Trimestre'], errors='coerce', format='mixed')

            if parsed.notna().any():
                df_largo['Orden_Trimestre'] = pd.to_datetime(df_largo['Trimestre']).rank(method='dense').astype(int)
            else:
                df_largo['Orden_Trimestre'] = pd.Categorical(df_largo['Trimestre'], categories=pd.unique(df_largo['Trimestre']), ordered=True).codes
        except Exception:
            df_largo['Orden_Trimestre'] = pd.Categorical(df_largo['Trimestre'], categories=pd.unique(df_largo['Trimestre']), ordered=True).codes
        df_largo = df_largo.sort_values(by=id_vars + ['Orden_Trimestre']).reset_index(drop=True)
        return df_largo

    try:
        df_largo = to_long(raw)
    except Exception as e:
        st.error(f"Error transformando a largo: {e}")
        st.stop()

    # helpers
    @st.cache_data
    def categories_list():
        return sorted(df_largo[nombre_col_categoria].dropna().unique().tolist())

    cats = categories_list()
    # Sidebar: selección global
    #st.sidebar.header('Controles globales')
    #sel_cats = st.sidebar.multiselect('Selecciona categorías (filtro global)', options=cats, default=cats[:6])
    #um_periods = st.sidebar.slider('Últimos N períodos', 3, 48, 12)
    with st.container(border=True):
        st.header('Controles globales')
        sel_cats = st.multiselect('Selecciona categorías (filtro global)', options=cats, default=cats[:6])
        um_periods = st.slider('Últimos N períodos', 3, 48, 12)

    # Tabs para A-E
    tabA, tabB, tabC, tabD, tabE = st.tabs(["A. Distribuciones & Probabilidades",
                                            "B. Clustering & Correlaciones",
                                            "C. Modelos Temporales & STL",
                                            "D. Visualizaciones Potentes",
                                            "E. Resumen / Dashboard" ])

    # ----------------------------
    # A: Distribuciones + probabilidades
    # ----------------------------
    with tabA:
        st.header("A. Distribuciones y probabilidades")
        col1, col2 = st.columns([2,1])
        with col2:
            sel_cat_A = st.selectbox('Elige una categoría para análisis de distribución', options=sel_cats)
            dist_choice = st.multiselect('Distribuciones a ajustar', ['normal','lognormal','gamma'], default=['normal','lognormal'])
            alpha = st.slider('Nivel para probabilidades (ej 0.05 -> 5%)', 0.01, 0.2, 0.05)
            threshold = st.number_input('Umbral para calcular probabilidad (ej 5.0)%', value=5.0)
        with col1:
            st.subheader(f"Distribución — {sel_cat_A}")
            data = df_largo[df_largo[nombre_col_categoria]==sel_cat_A]['Valor_Num']
            if data.empty:
                st.warning('No hay datos para la categoría seleccionada.')
            else:
                # Histograma + KDE (plotly)
                fig = px.histogram(data, x=data, nbins=30, marginal='box', histnorm='probability density', title=f"Histograma + KDE — {sel_cat_A}")
                # Ajustes de distribuciones
                info = {}
                try:
                    x = data.dropna().values
                    mu, sigma = stats.norm.fit(x)
                    info['normal'] = (mu, sigma)
                    # add normal pdf
                    xs = np.linspace(x.min(), x.max(), 200)
                    pdf = stats.norm.pdf(xs, mu, sigma)
                    fig.add_trace(go.Scatter(x=xs, y=pdf, mode='lines', name='Normal PDF'))
                except Exception as e:
                    st.warning('No se pudo ajustar normal: '+str(e))
                if 'lognormal' in dist_choice:
                    try:
                        # fit lognormal to positive data
                        xp = x[x>0]
                        if len(xp)>5:
                            shape, loc, scale = stats.lognorm.fit(xp, floc=0)
                            xs = np.linspace(xp.min(), xp.max(), 200)
                            pdf = stats.lognorm.pdf(xs, shape, loc=loc, scale=scale)
                            fig.add_trace(go.Scatter(x=xs, y=pdf, mode='lines', name='Lognormal PDF'))
                            info['lognormal'] = (shape, loc, scale)
                    except Exception as e:
                        st.warning('No se pudo ajustar lognormal: '+str(e))
                if 'gamma' in dist_choice:
                    try:
                        xp = x[x>0]
                        if len(xp)>5:
                            a, loc, scale = stats.gamma.fit(xp)
                            xs = np.linspace(xp.min(), xp.max(), 200)
                            pdf = stats.gamma.pdf(xs, a, loc=loc, scale=scale)
                            fig.add_trace(go.Scatter(x=xs, y=pdf, mode='lines', name='Gamma PDF'))
                            info['gamma'] = (a, loc, scale)
                    except Exception as e:
                        st.warning('No se pudo ajustar gamma: '+str(e))

                st.plotly_chart(fig, use_container_width=True)

                # QQ-plot + normality tests
                st.subheader('QQ-Plot y pruebas de normalidad')
                figqq, axqq = plt.subplots(figsize=(6,4))
                stats.probplot(x, plot=axqq)
                st.pyplot(figqq)

                shapiro_p = None
                try:
                    shapiro_p = stats.shapiro(x)[1]
                except Exception:
                    shapiro_p = np.nan
                try:
                    anderson = stats.anderson(x)
                    anderson_stat = anderson.statistic
                    anderson_crit = anderson.critical_values
                except Exception:
                    anderson_stat, anderson_crit = np.nan, None

                st.write({'shapiro_p': shapiro_p, 'anderson_stat': anderson_stat, 'anderson_crit': anderson_crit})

                # Probabilidades de escenarios
                st.subheader('Probabilidades (según ajuste normal por defecto)')
                mean_x = np.mean(x)
                std_x = np.std(x, ddof=1)
                prob_over = 1 - stats.norm.cdf(threshold, loc=mean_x, scale=std_x)
                prob_mean_plus_1sigma = 1 - stats.norm.cdf(mean_x + std_x, loc=mean_x, scale=std_x)

                st.markdown(f"- Media: **{mean_x:.3f}**, Desvío: **{std_x:.3f}**")
                st.markdown(f"- P(Valor > {threshold}) ≈ **{prob_over:.3f}**")
                st.markdown(f"- P(Valor > media + 1σ) ≈ **{prob_mean_plus_1sigma:.3f}**")

                # Probabilidad de volver a nivel pre-pandemia (ej: tomar promedio de 2018-2019)
                try:
                    # definir periodo pre-pandemia si las etiquetas son parseables
                    #parsed = pd.to_datetime(df_largo['Trimestre'], errors='coerce')
                    parsed = pd.to_datetime(df_largo['Trimestre'], errors='coerce', format='mixed')
                    if parsed.notna().any():
                        mask_pre = parsed.dt.year.isin([2018,2019])
                        pre_mean = df_largo.loc[mask_pre & (df_largo[nombre_col_categoria]==sel_cat_A),'Valor_Num'].mean()
                        prob_below_pre = stats.norm.cdf(pre_mean, loc=mean_x, scale=std_x)
                        st.markdown(f"- Promedio 2018-2019: **{pre_mean:.3f}** — P(Valor <= pre-pandemia) ≈ **{prob_below_pre:.3f}**")
                except Exception:
                    pass

    # ----------------------------
    # B: Clustering + correlaciones
    # ----------------------------
    with tabB:
        st.header("B. Clustering y correlaciones")
        st.write('Construimos características por categoría: media, volatilidad, pendiente (regresión), últimos valores')
        # construir features
        feats = []
        for name, g in df_largo.groupby(nombre_col_categoria):
            g = g.sort_values('Orden_Trimestre')
            vals = g['Valor_Num'].values
            mean = np.nanmean(vals)
            vol = np.nanstd(vals, ddof=1)
            # pendiente simple por regresión sobre orden
            if g['Orden_Trimestre'].nunique() >= 2:
                X = g['Orden_Trimestre'].values.reshape(-1,1)
                slope = np.polyfit(g['Orden_Trimestre'], vals, 1)[0]
            else:
                slope = 0
            last = vals[-1] if len(vals)>0 else np.nan
            feats.append((name, mean, vol, slope, last))
        df_feats = pd.DataFrame(feats, columns=[nombre_col_categoria,'mean','vol','slope','last'])
        df_feats = df_feats.dropna()
        st.dataframe(df_feats.set_index(nombre_col_categoria).round(3))

        # number of clusters
        k = st.slider('Número de clusters (k-means)', 2, min(10, len(df_feats)), 4)
        X = df_feats[['mean','vol','slope']].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Xs)
        df_feats['cluster'] = kmeans.labels_

        # mostrar mapa de clusters (scatter)
        figc = px.scatter(df_feats, x='mean', y='vol', color='cluster', hover_name=nombre_col_categoria, size='last', title='Clusters por media vs volatilidad')
        st.plotly_chart(figc, use_container_width=True)

        # dendrogram (hierarchical)
        st.subheader('Dendrograma (aglomerativo)')
        figd, axd = plt.subplots(figsize=(10,4))
        Z = hierarchy.linkage(Xs, method='ward')
        dn = hierarchy.dendrogram(Z, labels=df_feats[nombre_col_categoria].tolist(), ax=axd, leaf_rotation=90)
        st.pyplot(figd)

        # correlaciones entre categorías (matriz)
        st.subheader('Matriz de correlación entre series (por trimestre promedio)')
        pivot = df_largo.groupby([nombre_col_categoria, 'Trimestre'])['Valor_Num'].mean().reset_index()
        piv = pivot.pivot(index='Trimestre', columns=nombre_col_categoria, values='Valor_Num')
        corr = piv.corr()
        fig_corr = px.imshow(corr, labels=dict(x="Categoría", y="Categoría", color="Corr"), x=corr.columns, y=corr.index, aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)

    # ----------------------------
    # C: Modelos temporales + STL
    # ----------------------------
    with tabC:
        st.header('C. Modelos temporales y descomposición STL')
        sel_cat_C = st.selectbox('Selecciona categoría para modelado temporal', options=sel_cats, key='ts_cat')
        series = df_largo[df_largo[nombre_col_categoria]==sel_cat_C].sort_values('Orden_Trimestre')
        if series.empty:
            st.warning('No hay datos para categoría')
        else:
            ts = series.set_index('Orden_Trimestre')['Valor_Num']
            st.subheader('Descomposición STL')
            try:
                stl = STL(ts, period=4 if len(ts)>4 else 1, robust=True)
                res = stl.fit()
                fig_stl, ax = plt.subplots(3,1, figsize=(10,6), sharex=True)
                ax[0].plot(ts.index, res.trend); ax[0].set_title('Tendencia')
                ax[1].plot(ts.index, res.seasonal); ax[1].set_title('Estacional')
                ax[2].plot(ts.index, res.resid); ax[2].set_title('Residuo')
                st.pyplot(fig_stl)
            except Exception as e:
                st.warning('Error STL: '+str(e))

            st.subheader('ARIMA (configuración simple)')
            order = (1,0,1)
            try:
                model = ARIMA(ts, order=order).fit()
                st.write('ARIMA summary:')
                st.text(model.summary().as_text())
                # Forecast corto
                nfore = st.slider('Horizonte de forecast (periodos)', 1, 12, 4)
                fc = model.get_forecast(steps=nfore)
                mean_fc = fc.predicted_mean
                ci = fc.conf_int()
                figf = go.Figure()
                figf.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Hist'))
                figf.add_trace(go.Scatter(x=mean_fc.index, y=mean_fc.values, mode='lines', name='Forecast'))
                figf.add_trace(go.Scatter(x=ci.index.tolist()+ci.index[::-1].tolist(), y=ci.iloc[:,0].tolist()+ci.iloc[:,1][::-1].tolist(), fill='toself', name='CI', showlegend=False))
                st.plotly_chart(figf, use_container_width=True)
            except Exception as e:
                st.warning('Error ARIMA: '+str(e))

    # ----------------------------
    # D: Visualizaciones nuevas
    # ----------------------------
    with tabD:
        st.header('D. Visualizaciones interactivas')
        st.write('Varias opciones gráficas para explorar la estructura temporal y distribución')
        choice = st.selectbox('Elige visualización', ['Ridge plot (distribuciones por trimestre)', 'Boxplots por periodo', 'Calendar heatmap (por trimestre)', 'Sparklines interactivos'])
        if choice=='Ridge plot (distribuciones por trimestre)':
            # construimos distribuciones por trimestre para las categorías seleccionadas
            st.write("Ridge plot aproximado: densidades por trimestre (plotly)")
            pivot = df_largo[df_largo[nombre_col_categoria].isin(sel_cats)]
            # para cada trimestre calcular KDE
            num_periods = st.slider(
                "Cantidad de trimestres a incluir en el ridge plot",
                min_value=3,
                max_value=20,
                value=10
            )
            trimes = sorted(pivot['Trimestre'].unique(), key=lambda x: str(x))[-num_periods:]
            fig = go.Figure()
            ys = np.linspace(pivot['Valor_Num'].min(), pivot['Valor_Num'].max(), 200)
            for i,t in enumerate(trimes):
                vals = pivot[pivot['Trimestre']==t]['Valor_Num'].dropna().values
                if len(vals)<5: continue
                kde = stats.gaussian_kde(vals)
                dens = kde(ys)
                dens = dens / dens.max()  # normalizar para apilar
                fig.add_trace(go.Scatter(x=ys, y=dens + i*0.8, mode='lines', name=str(t), fill='toself'))
            fig.update_layout(height=400, yaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)

        if choice=='Boxplots por periodo':
            pivot = df_largo[df_largo[nombre_col_categoria].isin(sel_cats)]
            fig = px.box(pivot, x='Trimestre', y='Valor_Num', color=nombre_col_categoria, points='outliers')
            st.plotly_chart(fig, use_container_width=True)

        if choice=='Calendar heatmap (por trimestre)':
            pivot = df_largo[df_largo[nombre_col_categoria].isin(sel_cats)]
            # simple heatmap: categoría x trimestre
            table = pivot.groupby([nombre_col_categoria,'Trimestre'])['Valor_Num'].mean().reset_index()
            heat = table.pivot(index=nombre_col_categoria, columns='Trimestre', values='Valor_Num')
            fig = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index))
            st.plotly_chart(fig, use_container_width=True)

        if choice=='Sparklines interactivos':
            sample_items = df_largo[df_largo[nombre_col_categoria].isin(sel_cats)][nombre_col_item].dropna().unique().tolist()[:24]
            cols = st.columns(4)
            cnt = 0
            for item in sample_items:
                sub = df_largo[df_largo[nombre_col_item]==item].sort_values('Orden_Trimestre')
                fig = go.Figure(go.Scatter(x=sub['Trimestre'].tail(num_periods), y=sub['Valor_Num'].tail(num_periods), mode='lines'))
                cols[cnt%4].plotly_chart(fig, use_container_width=True)
                cnt += 1

    # ----------------------------
    # E: Todo junto (dashboard resumen)
    # ----------------------------
    with tabE:
        st.header('E. Dashboard resumen (todo junto)')
        st.write('KPI + mini visualizaciones — pensado para integrar a tu app principal como una de las pestañas')
        # KPI: top N categorías por valor reciente (dinámico)
        ultimo_trimestre = df_largo['Trimestre'].unique()[-1]
        df_reciente = df_largo[df_largo['Trimestre']==ultimo_trimestre]
        kpi = df_reciente.groupby(nombre_col_categoria)['Valor_Num'].mean().reset_index().sort_values('Valor_Num', ascending=False)

        # Mostrar KPIs en filas de N columnas (evita IndexError)
        num_kpi_cols = 3  # <--- Cambia aquí si querés más/menos columnas por fila
        kpi_to_show = kpi.head(9)  # mostrar hasta las primeras 9 por ejemplo (ajustable)
        if kpi_to_show.empty:
            st.info('No hay KPIs para mostrar (datos recientes vacíos).')
        else:
            kpi_list = kpi_to_show.reset_index(drop=True)
            for idx, row in kpi_list.iterrows():
                # cada vez que idx es múltiplo de num_kpi_cols creamos una nueva fila de columnas
                if idx % num_kpi_cols == 0:
                    kcols = st.columns(num_kpi_cols)
                col_idx = idx % num_kpi_cols
                kcols[col_idx].metric(label=str(row[nombre_col_categoria]), value=f"{row['Valor_Num']:.2f}")

        st.markdown('---')
        # mini: mapa de clusters
        with st.expander('Mini-clusters'):
            try:
                figc = px.scatter(df_feats, x='mean', y='vol', color='cluster', hover_name=nombre_col_categoria, size='last')
                st.plotly_chart(figc, use_container_width=True)
            except Exception:
                st.info('Mini-clusters no disponibles (df_feats no generado o vacío).')

        with st.expander('Mini-distribuciones (último trimestre)'):
            try:
                fig = px.histogram(df_reciente, x='Valor_Num', color=nombre_col_categoria, barmode='overlay', histnorm='probability density')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info('Mini-distribuciones no disponibles.')

        with st.expander('Anomalías (IsolationForest)'):
            # aplicar IsolationForest sobre pivot reciente
            pivot = df_largo.groupby([nombre_col_categoria, 'Trimestre'])['Valor_Num'].mean().reset_index()
            piv = pivot.pivot(index='Trimestre', columns=nombre_col_categoria, values='Valor_Num')
            #piv_f = piv.fillna(method='ffill').fillna(method='bfill')
            piv_f = piv.ffill().bfill()
            clf = IsolationForest(contamination=0.01, random_state=0)
            try:
                iso = clf.fit_predict(piv_f.T)
                anomalies = pd.DataFrame({'categoria': piv_f.columns, 'iso': iso})
                anomalies = anomalies[anomalies['iso']<0]
                st.write('Categorias anomalas detectadas:')
                st.write(anomalies)
            except Exception as e:
                st.warning('Error IsolationForest: '+str(e))

    st.markdown('---')


# --- Inportando librerias y codigo para acceder a botones de redireccionamiento a otras opciones de Menú ---
import base64
from pathlib import Path
from streamlit.components.v1 import html
import streamlit.components.v1 as components


# --- Inyectar CSS desde archivo ---
with open("assets/style.css","r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.write("###")
st.write("---")

with st.expander("Acceso a botones del Menú", expanded=False, icon=":material/auto_stories:", width="stretch"):
    # Centrar botones con columnas para acceso a otras páginas
    with st.container(border=True):
            col4, col5, col6 = st.columns([2, 2, 2])

            with col4:
                if st.button("Desocupación", key="btn", use_container_width=True):
                    st.switch_page("pages/2_desocupacion.py")

            with col5:    
                if st.button("Subocupación", key="btn-liquid", use_container_width=True):
                    st.switch_page("pages/3_subocupacion.py")

            with col6:
                if st.button("Informalidad", key="btn-glitch", use_container_width=True):
                    st.switch_page("pages/4_informalidad.py")


    # Centrar botones con columnas vacías a los lados
    with st.container(border=True):
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                pass  # espacio en blanco

            with col2:
                b1, b2 = st.columns(2)

                with b1:
                    if st.button("Inicio", key="main", use_container_width=True):
                        st.switch_page("main.py")

                with b2:
                    if st.button("Salir", key="exit", use_container_width=True):
                        st.switch_page("pages/5_salir.py")

            with col3:
                pass  # espacio en blanco


