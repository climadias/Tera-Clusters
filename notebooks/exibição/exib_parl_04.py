import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import sqlite3 as sql
import streamlit as st

import seaborn as sns
import csv
import squarify 

import os

#Dados de parlamentares
con = sql.connect("../dados/sql/base_completa.db")
df_parlamentares=pd.read_sql("select * from cadastro",con)
con.close()


#Dados de votação nos municípios
con = sql.connect("../dados/sql/base_completa.db")
df_municipios_votacao=pd.read_sql("select * from parl_votacao",con)
df_municipios=pd.read_sql("select * from municipios",con)
con.close()

# informação geográfica dos municípios
lista_uf=list(df_municipios_votacao["SG_UF"].unique())
df_mapa_brasil=pd.DataFrame()

# Monta mapa do Brasil inteiro
for uf in lista_uf:
    diretorio="../dados/ibge/shapes/"+uf
    arquivos=os.listdir(diretorio)
    shapefile = [arq for arq in arquivos if ".shp" in arq][0]
    arq=diretorio+"/"+shapefile
    df_uf= gpd.read_file(arq)
    df_uf["SG_UF"]=uf
    df_mapa_brasil=pd.concat([df_mapa_brasil,df_uf])

    
# ajusta colunas
df_mapa_brasil=df_mapa_brasil.astype({"CD_GEOCODM":"float64"})
df_mapa_brasil.rename(columns={"CD_GEOCODM":"CODIGO_IBGE"}, inplace=True)    

#Funções de exibição
# Plot estado e municipios com indicador
def plot_estado(estado,indicador,titulo, minimo,maximo, cor):
    vmin=minimo
    vmax=maximo
    fig, ax = plt.subplots(figsize=(12,8))
    estado.plot(ax=ax, column=indicador, cmap=cor, edgecolor="black", linewidth=0.2)
    sm = plt.cm.ScalarMappable(cmap=cor, norm=plt.Normalize(vmin=vmax, vmax=vmin))
    sm._A = []
    cbar = fig.colorbar(sm)
    plt.title(titulo, fontsize=20, color="grey")
    ax.axis('off')
    plt.axis("equal")
    plt.show()
    st.pyplot()
    return

def plot_treemap(estado,indicador,titulo, minimo,maximo, cor):
    estado["NOME_IDH"]=estado["NM_MUNICIPIO"]+"\n IDH - "+estado["IDHM_2010"].round(6).astype(str)+"\n Votos - "+(estado["PERC_VOTOS"]*100).round(2).astype(str)+"%"
    vmin=maximo
    vmax=minimo
    cmap = plt.cm.RdBu
    norm = plt.Normalize(vmin=vmax, vmax=vmin)
    colors = [cmap(norm(value)) for value in estado[indicador]]
    fig, ax = plt.subplots(figsize=(20,20))
    sm = plt.cm.ScalarMappable(cmap=cor, norm=plt.Normalize(vmin=vmax, vmax=vmin))
    sm._A = []
    #cbar = fig.colorbar(sm)
    plt.title(titulo, fontsize=20, color="grey")
    plt.axis('off')
    plt.axis("equal")
    squarify.plot(ax=ax, sizes=estado['VOTOS_TOTAL_MUN'][0:20], 
                  label=estado['NOME_IDH'][0:20], 
                  color=colors[0:20],
                  alpha=.8 )
    st.pyplot()
    return

def exibe_cadastro(cpf):
    df_parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    valor_bens=df_parlamentar["VL_BENS"]
    campos=["NM_URNA_CANDIDATO","NM_CANDIDATO","DT_NASCIMENTO","DS_COR_RACA",
            "DS_GRAU_INSTRUCAO","DS_OCUPACAO","VL_BENS"]
    parlamentar=df_parlamentar[campos].to_dict(orient="records")[0]
    #parlamentar["BENS"]=valor_bens.round(2)
    st.write(parlamentar)
    return

def exibe_votacao_estado(cpf, cpf_votacao):
    uf=cpf_votacao["SG_UF"].iloc[0]
    nome=cpf["NM_URNA_CANDIDATO"].iloc[0]
    cpf_municipios=df_mapa_brasil[df_mapa_brasil["SG_UF"]==uf]
    cpf_municipios=pd.merge(cpf_municipios,cpf_votacao, how="left", on="CODIGO_IBGE")
    cpf_municipios=pd.merge(cpf_municipios,df_municipios, how="left", on="CODIGO_IBGE")
    cpf_municipios=cpf_municipios.sort_values('VOTOS_TOTAL_MUN', ascending=False)
    indicador=indicador="IDHM_2010"
    print(cpf_municipios.columns)
    titulo=uf+" - "+"IDHM Municípios"
    minimo=0
    maximo=1
    cor="RdBu"
    plot_estado(cpf_municipios, indicador, titulo, minimo ,maximo, cor)
    indicador="VOTOS_TOTAL_MUN"
    titulo=nome+" - "+"Votos por município"
    minimo=0
    maximo=cpf_municipios[indicador].max()
    cor="Blues"
    plot_estado(cpf_municipios, indicador, titulo, minimo ,maximo, cor)
    indicador=indicador="IDHM_2010"
    titulo=nome+" - "+"IDHM e Votos por município"
    minimo=0
    maximo=1
    cor="RdBu"
    plot_treemap(cpf_municipios, indicador, titulo, minimo ,maximo, cor)
    return


lista=list(df_parlamentares["SG_UE"].unique())
lista_uf=sorted(lista)
uf=lista_uf[0]

st.title("Parlamentar")
#st.sidebar.write("Selecione o estado")

uf = st.sidebar.selectbox("Selecione o Estado",lista_uf)
st.write("Estado {}".format(uf))

lista=list(df_parlamentares[df_parlamentares["SG_UE"]==uf]["NM_CANDIDATO"])
lista_parlamentar=sorted(lista)
nome=lista_parlamentar[0]

nome = st.sidebar.selectbox("selecione o parlamentar",lista_parlamentar)
st.write("Parlamentar {}".format(nome))

cpf=df_parlamentares[df_parlamentares["NM_CANDIDATO"]==nome]["CPF"].iloc[0]

df_cpf_votacao=df_municipios_votacao[df_municipios_votacao["CPF"]==cpf]
df_cpf=df_parlamentares[df_parlamentares["CPF"]==cpf]
exibe_cadastro(cpf)
exibe_votacao_estado(df_cpf, df_cpf_votacao)
