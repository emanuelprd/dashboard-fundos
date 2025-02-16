#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
import gdown

#%% Streamlit Configuração
st.set_page_config(
    page_title="Dashboard de Fundos",
    page_icon="📊",
    layout="wide"  
)

#%% Importando a base de fundos
@st.cache_data
def carregar_base():
    url = 'https://drive.google.com/file/d/1tHCk4Lh4Tovi5ugKMM4jN1pvaEMJxOdu/view?usp=sharing'
    output = 'base_fundos.csv'
    gdown.download(url, output, quiet=False)
    
    chunks = []
    for chunk in pd.read_csv("base_fundos.csv", sep=',', chunksize=1000):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)    
    df.set_index("DT_COMPTC", inplace=True)
    
    return df

df_fundos_adj = carregar_base()

#%% Definindo as funções

def fundos_graficos(colunas, rac, boxplot, drawdown, vol):
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Retorno Histórico (Diário)", "Boxplot dos Retornos (Mensal)", "Drawdown (Diário)", "Volatilidade Móvel (Mensal)"),
                        vertical_spacing=0.2,
                        shared_xaxes=False,
                        shared_yaxes=False)
    
    for num, col in enumerate(colunas):
        
        legend_group = f'group_{col}'
        
        # Retorno Histórico
        fig.add_trace(go.Scatter(x=rac.index, y=rac[col], mode='lines', name=col, line=dict(color=cores_fundos[num]), showlegend=True, legendgroup=legend_group),
                      row=1, col=1)
        
        # Boxplot dos Retornos    
        fig.add_trace(go.Box(y=boxplot[col], name=col, line=dict(color=cores_fundos[num]), showlegend=False, legendgroup=legend_group),
                      row=1, col=2)
        # Drawdown    
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown[col], mode='lines', name=col, line=dict(color=cores_fundos[num]), showlegend=False, legendgroup=legend_group),
                      row=2, col=1)
        
        # Volatilidade Móvel    
        fig.add_trace(go.Scatter(x=vol.index, y=vol[col], mode='lines', name=col, line=dict(color=cores_fundos[num]), showlegend=False, legendgroup=legend_group),
                      row=2, col=2)
        
    fig.update_layout(
        title="Análise Gráfica",
        height=900,
        showlegend=True)
    
    st.plotly_chart(fig)
    
def fundos_correlacao(rd):
    corr_fig = go.Figure(data=go.Heatmap(
        z=rd.corr(),
        x=rd.columns,
        y=rd.columns,
        colorscale='Blues',
        colorbar=dict(title="Correlação"),
        text=rd.corr().round(2).values,
        texttemplate="%{text}",
        hoverinfo='none'))

    corr_fig.update_layout(
        title="Matriz de Correlação",
        height=600,
        width=700)
    
    st.plotly_chart(corr_fig)

#%% Dashboard

st.title("📊 Dashboard de Fundos de Investimento")

st.subheader("Insira o CNPJ e Apelido do Fundo")

# Inicializa listas na session_state
if "nome_fundos_input" not in st.session_state:
    st.session_state.nome_fundos_input = [""] * 5
if "cnpj_fundos_input" not in st.session_state:
    st.session_state.cnpj_fundos_input = [""] * 5

# Layout para inputs
col1, col2 = st.columns(2)

with col1:
    for i in range(5):
        st.session_state.nome_fundos_input[i] = st.text_input(
            f"Apelido do Fundo {i+1}", 
            value=st.session_state.nome_fundos_input[i], 
            key=f"nome_fundo_{i}"
        )

with col2:
    for i in range(5):
        st.session_state.cnpj_fundos_input[i] = st.text_input(
            f"CNPJ do Fundo {i+1}", 
            value=st.session_state.cnpj_fundos_input[i], 
            key=f"cnpj_fundo_{i}"
        )

#%% Gerar as Estatísticas

# Botão para pesquisar
if st.button("🔎 Buscar Fundo"):
    fundos_consulta_dict = {}
    
    for nome, cnpj in zip(st.session_state.nome_fundos_input, st.session_state.cnpj_fundos_input):
        if nome and cnpj:
            df_filtrado = df_fundos_adj[df_fundos_adj["CNPJ_FUNDO"] == cnpj]
            
            if df_filtrado.empty:
                st.error(f"⚠️ Nenhum fundo encontrado para o CNPJ: {cnpj}")
            else:
                fundos_consulta_dict[nome] = df_filtrado['VL_QUOTA']

    if fundos_consulta_dict:
        fundos_consulta = pd.DataFrame(fundos_consulta_dict)
  
        # Retorno Diário
        rd = fundos_consulta.pct_change() 
        
        # Retorno Mensal
        rm = fundos_consulta.resample("ME").last().pct_change()
        
        # Cota Mensal
        cm = fundos_consulta.resample("ME").last()
        
        # Retorno Acumulado
        rac = (fundos_consulta / fundos_consulta.iloc[0] - 1) 
        
        # Drawdown
        r_acumulado = (1 + rd).cumprod()
        rd_drawdown = r_acumulado / r_acumulado.cummax() - 1
        
        # Volatilidade Móvel
        rd_vol = (rd.rolling(30).std() * np.sqrt(252)).dropna()
        
        # Volatilidade Anual
        vol_anual = rd.std() * np.sqrt(252) * 100

        # Retorno Composto
        periodo = np.busday_count(rac.index[0].strftime('%Y-%m-%d'), rac.index[-1].strftime('%Y-%m-%d'))
        r_composto = (rac.iloc[-1] + 1) ** (1 / (periodo / 252)) - 1
    
        rentabilidade_resumo = pd.DataFrame({
            "Retorno Acumulado (%)": rac.iloc[-1] * 100,
            "Retorno Composto (%)": r_composto * 100,
            "Retorno 12M (%)": ((cm/cm.shift(12)-1)*100).iloc[-1],
            "Retorno 24M (%)": ((cm/cm.shift(24)-1)*100).iloc[-1],
            "Retorno 36M (%)": ((cm/cm.shift(36)-1)*100).iloc[-1]
            }).T
        
        volatilidade_resumo = pd.DataFrame({
            "Volatilidade Anual Média (%)": vol_anual,
            "VaR (5%)": stats.norm.ppf(0.05, rm.mean(), rm.std())*100,
            "Risco - Retorno (%)": (r_composto / vol_anual)*100,
            "Meses Negativos": rm[rm<0].count(),
            "Meses Positivos": rm[rm>0].count(),
            "Maior Drawdown (%)": rd_drawdown.min() * 100
            }).T

        cores_fundos = ["blue","green","red","orange","purple"]

        # Gráficos 2x2
        fundos_graficos(list(fundos_consulta.columns), rac, rm, rd_drawdown, rd_vol)
        
        # Heatmap de Correlação
        fundos_correlacao(rd)

        # Exibir Resumo Estatístico
        st.subheader("Resumo Rentabilidade (%)")
        st.dataframe(rentabilidade_resumo.style.set_properties(**{"background-color": "#f5f5f5", "border-color": "black"}).format("{:.2f}"))
        
        st.subheader("Resumo Volatilidade (%)")
        st.dataframe(volatilidade_resumo.style.set_properties(**{"background-color": "#f5f5f5", "border-color": "black"}).format("{:.2f}"))
        
        st.subheader("Retornos Mensais (%)")
        st.dataframe((rm*100).describe().style.set_properties(**{"background-color": "#f5f5f5", "border-color": "black"}).format("{:.2f}"))
        
        st.download_button(
            label="📁 Baixar Base de Fundos",
            data=fundos_consulta.to_csv(),
            file_name="fundos_consulta.csv",
            mime="text/csv")
    else:
        st.error("Nenhum fundo válido foi encontrado. Verifique os CNPJs inseridos.")
