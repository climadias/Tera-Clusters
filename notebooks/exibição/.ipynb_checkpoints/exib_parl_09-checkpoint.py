import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import sqlite3 as sql
import streamlit as st

import locale

import seaborn as sns
import csv
import squarify 

import plotly.express as px
import plotly.offline as pyo

import os

#Dados de parlamentares
con = sql.connect("../../dados/sql/base_completa.db")
df_parlamentares=pd.read_sql("select * from cadastro",con)
df_parl_bens=pd.read_sql("select * from parl_bens",con)
df_deputados=pd.read_sql("select * from kpi_cadastro_basico",con)
df_gastos_gabinete=pd.read_sql("select * from deputados_gastos_gabinete",con,parse_dates=["DT_GASTO"])
con.close()

df_parlamentares=df_parlamentares[df_parlamentares["DS_CARGO"]=="DEPUTADO FEDERAL"].copy()

df_deputados=df_deputados[['CPF', 'ID_CAMARA', 'NM_PUBLICO','ORGAO_PARTICIPANTE',
                     'ORGAO_TOTAL', 'ORGAO_GESTOR', 'PERC_PRESENCA', 
                     'TOTAL_PROPOSTAS', 'GASTO_GABINETE']].copy()

# Tratamento de deputado com 2 ID_CAMARA!!!
df_deputados.drop_duplicates(inplace=True)
df_deputados=df_deputados[df_deputados["ID_CAMARA"]!="131410"]
df_deputados.reset_index(drop=True,inplace=True)


df_parlamentares=pd.merge(df_parlamentares,df_deputados, how="left", on="CPF")

#### 
# Inserção de informação de cluster
#
df_cluster=pd.read_csv("./ALO.csv", dtype={"CPF":"object"}, sep=";")
cpf=df_cluster["CPF"]=="00558574181"
clu=df_cluster["kmeans"]==2
df_cluster=df_cluster[~(cpf & clu)]
df_cluster.rename(columns={"kmeans":"CLUSTER"}, inplace=True)
df_cluster=df_cluster[["CPF","CLUSTER"]].copy()
df_cluster.drop_duplicates(inplace=True)
df_cluster.reset_index(drop=True, inplace=True)

df_parlamentares=pd.merge(df_parlamentares,df_cluster, how="left", on="CPF")


df_original=df_parlamentares.copy()

# carga das métricas de cluster
#["GASTO_GABINETE","TOTAL_PROPOSTAS","PERC_PRESENCA","ORGAO_GESTOR","ORGAO_PARTICIPANTE","followers_count"]
df_cluster_metricas=pd.read_csv("./var_escala0a10_kmeans.csv",sep=";")
df_cluster_metricas.rename(columns={"Presença(%)":"PERC_PRESENCA",
                                   "Twitter":"followers_count",
                                   "Gastos":"GASTO_GABINETE",
                                   "Propostas":"TOTAL_PROPOSTAS",
                                   "Participação em órgãos":"ORGAO_PARTICIPANTE",
                                   "Gestão em órgãos":"ORGAO_GESTOR",
                                   "Unnamed: 0":"CLUSTER"}, inplace=True)


#Dados de votação nos municípios
con = sql.connect("../../dados/sql/base_completa.db")
df_municipios_votacao=pd.read_sql("select * from parl_votacao",con)
df_municipios=pd.read_sql("select * from municipios",con)
con.close()

# informação geográfica dos municípios
lista_uf=list(df_municipios_votacao["SG_UF"].unique())
df_mapa_brasil=pd.DataFrame()

# Monta mapa do Brasil inteiro
for uf in lista_uf:
    diretorio="../../dados/ibge/shapes/"+uf
    arquivos=os.listdir(diretorio)
    shapefile = [arq for arq in arquivos if ".shp" in arq][0]
    arq=diretorio+"/"+shapefile
    df_uf= gpd.read_file(arq)
    df_uf["SG_UF"]=uf
    df_mapa_brasil=pd.concat([df_mapa_brasil,df_uf])

    
# ajusta colunas
df_mapa_brasil.rename(columns={"CD_GEOCODM":"CODIGO_IBGE"}, inplace=True)
df_mapa_brasil=df_mapa_brasil.astype({"CODIGO_IBGE":"float64"})
df_municipios=df_municipios.astype({"CODIGO_IBGE":"float64"})
df_municipios_votacao=df_municipios_votacao.astype({"CODIGO_IBGE":"float64"})

# Ajusta mudança Rafael
df_municipios.drop(columns=["VOTOS_TOTAL_MUN","DS_CARGO","CPF"], inplace=True)
df_municipios.drop_duplicates(inplace=True)
df_municipios.reset_index(drop=True, inplace=True)
df_municipios_votacao.drop(columns=["NM_MUNICIPIO","DS_CARGO",
                                    "VOTOS_TOTAL_PARL","VOTOS_TOTAL_EST",
                                    "PERC_VOTOS_EST"], inplace=True)
df_municipios_votacao.drop_duplicates(inplace=True)
df_municipios_votacao.reset_index(drop=True,inplace=True)
df_municipios_votacao.rename(columns={"PERC_VOTOS_INDIVIDUAL":"PERC_VOTOS"},inplace=True)


#Funções de exibição
# Plot estado e municipios com indicador
def plot_estado(estado,indicador,titulo, minimo,maximo, cor):
    st.subheader(titulo)
    vmin=minimo
    vmax=maximo
    norm=plt.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(12,8))
    estado.plot(ax=ax, column=indicador, cmap=cor, norm=norm, edgecolor="black", linewidth=0.2)
    #sm = plt.cm.ScalarMappable(cmap=cor, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm = plt.cm.ScalarMappable(cmap=cor, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    #plt.title(titulo, fontsize=20, color="grey")
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

def plot_px_treemap(estado,indicador,titulo, minimo,maximo, cor):
    estado["NOME_IDH"]=estado["NM_MUNICIPIO"]+"\n IDH - "+estado["IDHM_2010"].round(6).astype(str)+"\n Votos - "+(estado["PERC_VOTOS"]*100).round(2).astype(str)+"%"
    st.subheader(titulo)
    vmin=maximo
    vmax=minimo
    estado=estado[estado["VOTOS_TOTAL_MUN"]>0]
    fig = px.treemap(estado, path=['SG_UF', 'NM_MUNICIPIO'], values='VOTOS_TOTAL_MUN',
                  color='IDHM_2010', hover_data=['PERC_VOTOS'],
                  color_continuous_scale='RdBu',
                  range_color=[0,1])
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_cadastro(cpf):
    st.header("Informações básicas")
    df_parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    valor_bens=df_parlamentar["VL_BENS"]
    campos=["NM_CANDIDATO","NM_URNA_CANDIDATO","DS_GENERO","DS_COR_RACA","DT_NASCIMENTO",
            "DS_GRAU_INSTRUCAO","DS_OCUPACAO","DS_ESTADO_CIVIL","SG_PARTIDO","ANO_ELEICAO","CLUSTER"]
    #parlamentar=df_parlamentar[campos].to_dict(orient="records")[0]
    parlamentar=df_parlamentar[campos]
    parlamentar.rename({"NM_CANDIDATO": "Nome","NM_URNA_CANDIDATO":"Nome na Urna",
                       "DT_NASCIMENTO":"Nascimento","DS_COR_RACA":"Cor",
                        "DS_GENERO":"Gênero","DS_ESTADO_CIVIL":"Estado Civil",
                        "DS_OCUPACAO":"Ocupação ant.","SG_PARTIDO":"Partido",
                        "DS_CARGO":"Cargo","ANO_ELEICAO":"Eleito em",
                       "DS_GRAU_INSTRUCAO":"Grau de instrução","CLUSTER":"Perfil de atuação"}, inplace=True, axis=1)
    #parlamentar["BENS"]=valor_bens.round(2)
    st.dataframe(parlamentar.T)
    return

def exibe_votacao_estado(cpf, cpf_votacao):
    st.header("Distribuição geográfica da votação")
    uf=cpf_votacao["SG_UF"].iloc[0]
    nome=cpf["NM_URNA_CANDIDATO"].iloc[0]
    cpf_municipios=df_mapa_brasil[df_mapa_brasil["SG_UF"]==uf]
    cpf_municipios=pd.merge(cpf_municipios,cpf_votacao, how="left", on="CODIGO_IBGE")
    cpf_municipios=pd.merge(cpf_municipios,df_municipios, how="left", on="CODIGO_IBGE")
    cpf_municipios=cpf_municipios.sort_values('VOTOS_TOTAL_MUN', ascending=False)
    indicador=indicador="IDHM_2010"
    print(cpf_municipios.columns)
    titulo="IDHM Municípios"+" - "+uf
    minimo=0.0
    maximo=1.0
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
    plot_px_treemap(cpf_municipios, indicador, titulo, minimo ,maximo, cor)
    return

def exibe_genero(cpf):
    st.subheader("Considerando o gênero")
    parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    genero=parlamentar["DS_GENERO"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    gen_brasil=pd.crosstab(index=[df_parlamentares['DS_GENERO']], columns = [df_parlamentares['DS_GENERO']], 
            normalize=True).T
    perc_brasil=(gen_brasil[genero][genero])*100
    gen_estado=pd.crosstab(index=[df_parlamentares['DS_GENERO']], columns = [df_parlamentares[df_parlamentares["SG_UE"]==uf]['DS_GENERO']], 
            normalize=True).T
    perc_estado=(gen_estado[genero][genero])*100
    st.write("Faz parte dos {}% dos parlamentares do sexo {} no Brasil e {}% do estado {}.".format(perc_brasil.round(1),
                                                                                                   genero.lower(),
                                                                                                   perc_estado.round(1),
                                                                                                   uf))
    df_genero=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","DS_GENERO"]].copy()
    df_genero=df_genero[df_genero["SG_UE"]==uf].copy()
    data = df_genero
    fig =px.sunburst(data,
                path=['SG_UE', "DS_GENERO","NM_PUBLICO"])
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_cor(cpf):
    st.subheader("Considerando a cor declarada")
    parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    cor=parlamentar["DS_COR_RACA"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    x_brasil=pd.crosstab(index=[df_parlamentares['DS_COR_RACA']], columns = [df_parlamentares['DS_COR_RACA']], 
            normalize=True).T
    perc_brasil=x_brasil[cor][cor].round(3)*100
    x_estado=pd.crosstab(index=[df_parlamentares['DS_COR_RACA']], columns = [df_parlamentares[df_parlamentares["SG_UE"]==uf]['DS_COR_RACA']], 
            normalize=True).T
    perc_estado=x_estado[cor][cor].round(3)*100
    st.write("Faz parte dos {}% dos parlamentares de cor {} no Brasil e {}% do estado {}.".format(perc_brasil.round(1),
                                                                                                  cor.lower(),
                                                                                                  perc_estado.round(1),
                                                                                                  uf))
    df_cor=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","DS_COR_RACA"]].copy()
    df_cor=df_cor[df_cor["SG_UE"]==uf].copy()
    data = df_cor
    fig =px.sunburst(data,
                path=['SG_UE', "DS_COR_RACA","NM_PUBLICO"])
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_estado_civil(cpf):
    st.subheader("Considerando o Estado Civil")
    parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    civil=parlamentar["DS_ESTADO_CIVIL"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    x_brasil=pd.crosstab(index=[df_parlamentares['DS_ESTADO_CIVIL']], columns = [df_parlamentares['DS_ESTADO_CIVIL']], 
            normalize=True).T
    perc_brasil=x_brasil[civil][civil].round(3)*100
    x_estado=pd.crosstab(index=[df_parlamentares['DS_ESTADO_CIVIL']], columns = [df_parlamentares[df_parlamentares["SG_UE"]==uf]['DS_ESTADO_CIVIL']], 
            normalize=True).T
    perc_estado=x_estado[civil][civil].round(3)*100
    st.write("Faz parte dos {}% dos parlamentares com estado civil {} no Brasil, {}% no estado {}.".format(perc_brasil.round(1),
                                                                                                           civil.lower(),
                                                                                                           perc_estado.round(1)
                                                                                                           ,uf))
    df_civil=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","DS_ESTADO_CIVIL"]].copy()
    df_civil=df_civil[df_civil["SG_UE"]==uf].copy()
    data = df_civil
    fig =px.sunburst(data,
                path=['SG_UE',"DS_ESTADO_CIVIL","NM_PUBLICO"])
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_instrucao(cpf):
    st.subheader("Considerando o grau de instrução")
    parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    grau=parlamentar["DS_GRAU_INSTRUCAO"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    x_brasil=pd.crosstab(index=[df_parlamentares['DS_GRAU_INSTRUCAO']], columns = [df_parlamentares['DS_GRAU_INSTRUCAO']], 
            normalize=True).T
    perc_brasil=x_brasil[grau][grau].round(3)*100
    x_estado=pd.crosstab(index=[df_parlamentares['DS_GRAU_INSTRUCAO']], columns = [df_parlamentares[df_parlamentares["SG_UE"]==uf]['DS_GRAU_INSTRUCAO']], 
            normalize=True).T
    perc_estado=x_estado[grau][grau].round(3)*100
    st.write("Faz parte dos {}% dos parlamentares com {} no Brasil, {}% no estado {}.".format(perc_brasil.round(1),
                                                                                              grau.lower(),
                                                                                              perc_estado.round(1),
                                                                                              uf))
    df_instruc=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","DS_GRAU_INSTRUCAO"]].copy()
    df_instruc=df_instruc[df_instruc["SG_UE"]==uf].copy()
    data = df_instruc
    fig =px.sunburst(data,
                path=['SG_UE',"DS_GRAU_INSTRUCAO","NM_PUBLICO"])
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_ocupacao(cpf):
    st.subheader("Considerando a acupação anterior")
    parlamentar=df_parlamentares[df_parlamentares["CPF"]==cpf]
    ocupacao=parlamentar["DS_OCUPACAO"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    x_brasil=pd.crosstab(index=[df_parlamentares['DS_OCUPACAO']], columns = [df_parlamentares['DS_OCUPACAO']], 
            normalize=True).T
    perc_brasil=x_brasil[ocupacao][ocupacao].round(3)*100
    x_estado=pd.crosstab(index=[df_parlamentares['DS_OCUPACAO']], columns = [df_parlamentares[df_parlamentares["SG_UE"]==uf]['DS_OCUPACAO']], 
            normalize=True).T
    perc_estado=x_estado[ocupacao][ocupacao].round(3)*100
    st.write("Faz parte dos {}% dos parlamentares com ocupação anterior de {} no Brasil, {}% no estado {}.".format(perc_brasil.round(1),
                                                                                                                   ocupacao.lower(),
                                                                                                                   perc_estado.round(1),
                                                                                                                   uf))
    df_ocupacao=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","DS_OCUPACAO"]].copy()
    df_ocupacao=df_ocupacao[df_ocupacao["SG_UE"]==uf].copy()
    data = df_ocupacao
    fig =px.sunburst(data,
                path=['SG_UE',"DS_OCUPACAO","NM_PUBLICO"])
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return


def exibe_bens(cpf):
    st.header("Bens do parlamentar")
    loc=locale.setlocale( locale.LC_ALL, '' )
    df_bens=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","VL_BENS"]].copy()
    df_bens["BENS_BR"]=df_bens["VL_BENS"].rank(ascending=False)
    df_bens["BENS_UF"]=df_bens.groupby("SG_UE")["VL_BENS"].rank(ascending=False)
    df_bens["BENS_PARTIDO"]=df_bens.groupby("SG_PARTIDO")["VL_BENS"].rank(ascending=False)
    parlamentar=df_bens[df_bens["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    vl_bens=parlamentar["VL_BENS"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    vl_bens=locale.currency( vl_bens, grouping=True )
    rank_br=int(parlamentar["BENS_BR"].iloc[0])
    rank_estado=int(parlamentar["BENS_UF"].iloc[0])
    rank_partido=int(parlamentar["BENS_PARTIDO"].iloc[0])
    st.write("Com bens declarados no valor de {}, no Brasil seria o é o {}o. colocado. no estado  {} o {}o.".format(vl_bens,
                                                                                                                   rank_br,
                                                                                                                   uf,
                                                                                                                   rank_estado))
    st.write("Dentro do partido {}, é o {}o.".format(partido, rank_partido))
    lista_bens=df_parl_bens.copy()
    lista_bens=lista_bens.drop(columns=["VR_DESPESA_CONTRATADA","CD_TIPO_BEM_CANDIDATO"])
    lista_bens=lista_bens[lista_bens["CPF"]==cpf].groupby("DS_TIPO_BEM_CANDIDATO").sum()
    lista_bens=lista_bens.sort_values("VR_BEM_CANDIDATO", ascending=False)
    lista_bens["VR_BEM_CANDIDATO"]=lista_bens["VR_BEM_CANDIDATO"].apply(lambda x:locale.currency(x, grouping=True ))
    lista_bens.rename(columns={"VR_BEM_CANDIDATO":"Valor"}, inplace=True)
    lista_bens.index.name="Tipo de bem"
    st.write(lista_bens)
    #uf=parlamentar["SG_UE"].iloc[0]
    df_bens=df_bens[df_bens["SG_UE"]==uf].copy()
    data = df_bens
    fig =px.treemap(
    data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values='VL_BENS')
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_twitter(cpf):
    st.header("Seguidores Twitter")
    df_twitter=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","followers_count"]].copy()
    df_twitter.fillna(0, inplace=True)
    df_twitter["FOLL_BR"]=df_twitter["followers_count"].rank(ascending=False)
    df_twitter["FOLL_UF"]=df_twitter.groupby("SG_UE")["followers_count"].rank(ascending=False)
    df_twitter["FOLL_PARTIDO"]=df_twitter.groupby("SG_PARTIDO")["followers_count"].rank(ascending=False)
    parlamentar=df_twitter[df_twitter["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    followers=parlamentar["followers_count"].iloc[0]
    loc=locale.setlocale( locale.LC_ALL, '' )
    followers=locale.str(followers)
    rank_br=int(parlamentar["FOLL_BR"].iloc[0])
    rank_estado=int(parlamentar["FOLL_UF"].iloc[0])
    rank_partido=int(parlamentar["FOLL_PARTIDO"].iloc[0])
    st.write("Com {} seguidores no Twiter, no Brasil é {}o. colocado. no estado {} é o {}o".format(followers,
                                                                                                rank_br,
                                                                                                uf,
                                                                                                rank_estado))
    st.write("Dentro do partido {}, é o {}o.".format(partido, rank_partido))
    df_twitter=df_twitter[df_twitter["SG_UE"]==uf].copy()
    data = df_twitter
    fig =px.treemap(
        data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values='followers_count')
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_gastos_campanha(cpf):
    st.header("Gastos de campanha")
    df_gastos=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","VR_DESPESA_CONTRATADA"]].copy()
    df_gastos["BR"]=df_gastos["VR_DESPESA_CONTRATADA"].rank(ascending=False)
    df_gastos["UF"]=df_gastos.groupby("SG_UE")["VR_DESPESA_CONTRATADA"].rank(ascending=False)
    df_gastos["PARTIDO"]=df_gastos.groupby("SG_PARTIDO")["VR_DESPESA_CONTRATADA"].rank(ascending=False)
    parlamentar=df_gastos[df_gastos["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    gastos=parlamentar["VR_DESPESA_CONTRATADA"].iloc[0]
    gastos=locale.currency(gastos, grouping=True )
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])
    st.write("Com gastos declarados de {}, no Brasil foi o {}o. colocado em custos de campanha, no estado {} foi o {}o".format(gastos,
                                                                                                  rank_br,
                                                                                                  uf,
                                                                                                  rank_estado))
    st.write("Dentro do partido {}, foi o {}o.".format(partido, rank_partido))
    df_gastos=df_gastos[df_gastos["SG_UE"]==uf].copy()
    data = df_gastos
    fig =px.treemap(
        data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values='VR_DESPESA_CONTRATADA')
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_votacao(cpf):
    st.header("Votação")
    df_votos=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","VOTOS_TOTAL_PARL"]].copy()
    df_votos["BR"]=df_votos["VOTOS_TOTAL_PARL"].rank(ascending=False)
    df_votos["UF"]=df_votos.groupby("SG_UE")["VOTOS_TOTAL_PARL"].rank(ascending=False)
    df_votos["PARTIDO"]=df_votos.groupby("SG_PARTIDO")["VOTOS_TOTAL_PARL"].rank(ascending=False)
    parlamentar=df_votos[df_votos["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    votacao=parlamentar["VOTOS_TOTAL_PARL"].iloc[0]
    votacao=locale.str(votacao)
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])
    st.write("Com {} votos, no Brasil foi o {}o mais votado, no {} foi o {}o".format(votacao,
                                                                              rank_br,
                                                                              uf,
                                                                              rank_estado))
    st.write("Dentro do partido {}, foi o {}o.".format(partido,rank_partido))
    df_votos=df_votos[df_votos["SG_UE"]==uf].copy()
    data = df_votos
    fig =px.treemap(
        data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values='VOTOS_TOTAL_PARL')
    st.plotly_chart(fig, use_container_width=True)
    return
    #
    

def exibe_assiduidade(cpf):
    st.header("Presença em sessões deliberativas")
    df_freq=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","PERC_PRESENCA"]].copy()
    df_freq["BR"]=df_freq["PERC_PRESENCA"].rank(ascending=False)
    df_freq["UF"]=df_freq.groupby("SG_UE")["PERC_PRESENCA"].rank(ascending=False)
    df_freq["PARTIDO"]=df_freq.groupby("SG_PARTIDO")["PERC_PRESENCA"].rank(ascending=False)
    parlamentar=df_freq[df_freq["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    freq=parlamentar["PERC_PRESENCA"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    freq=locale.str(freq)
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])

    st.write("Com {}% de assiduidade, no Brasil é o {}o mais assíduo, no {} seria o {}o".format(freq,
                                                                              rank_br,
                                                                              uf,
                                                                              rank_estado))
    st.write("Dentro do partido {}, seria o {}o.".format(partido,rank_partido))
    df_freq=df_freq[df_freq["SG_UE"]==uf].copy()
    df_freq.sort_values("PERC_PRESENCA", ascending=True, inplace=True)
    df_freq.rename(columns={"PERC_PRESENCA":"Presença em sessões deliberativas",
                        "NM_PUBLICO":"Nome",
                       "SG_PARTIDO":"Partido"}, inplace=True)
    data = df_freq
    fig =px.bar(
        data,x="Presença em sessões deliberativas",y="Nome",color="Partido",
        hover_data=["Nome","Presença em sessões deliberativas"],
        orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_propostas(cpf):
    st.header("Propostas apresentadas")
    df_prop=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","TOTAL_PROPOSTAS"]].copy()
    df_prop["BR"]=df_prop["TOTAL_PROPOSTAS"].rank(ascending=False)
    df_prop["UF"]=df_prop.groupby("SG_UE")["TOTAL_PROPOSTAS"].rank(ascending=False)
    df_prop["PARTIDO"]=df_prop.groupby("SG_PARTIDO")["TOTAL_PROPOSTAS"].rank(ascending=False)
    parlamentar=df_prop[df_prop["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    prop=parlamentar["TOTAL_PROPOSTAS"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    prop=locale.str(prop)
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])
    st.write("Com {} propostas, no Brasil é o {}o com mais propostas, no {} seria o {}o".format(prop,
                                                                              rank_br,
                                                                              uf,
                                                                              rank_estado))
    st.write("Dentro do partido {}, seria o {}o.".format(partido,rank_partido))
    df_prop=df_prop[df_prop["SG_UE"]==uf].copy()
    df_prop.sort_values("TOTAL_PROPOSTAS", ascending=True, inplace=True)
    #df_prop.rename(columns={"TOTAL_PROPOSTAS":"Total de propostas",
    #                        "NM_PUBLICO":"Nome"}, inplace=True)
    data = df_prop
    fig =px.treemap(
    data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values="TOTAL_PROPOSTAS")
    st.plotly_chart(fig, use_container_width=True)
    return


def exibe_gestao(cpf):
    st.header("Participações como gestor de orgãos ou comissões da Câmara")
    df_part_g=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","ORGAO_GESTOR"]].copy()
    df_part_g.fillna(0, inplace=True)
    df_part_g["BR"]=df_part_g["ORGAO_GESTOR"].rank(ascending=False)
    df_part_g["UF"]=df_part_g.groupby("SG_UE")["ORGAO_GESTOR"].rank(ascending=False)
    df_part_g["PARTIDO"]=df_part_g.groupby("SG_PARTIDO")["ORGAO_GESTOR"].rank(ascending=False)
    parlamentar=df_part_g[df_part_g["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    part_g=parlamentar["ORGAO_GESTOR"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    part_g=locale.str(part_g)
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])
    st.write("Com {} participação como gestor de orgão ou comissão, no Brasil é o {}o com mais cargos de gestão, no {} seria o {}o".format(part_g,
                                                                              rank_br,
                                                                              uf,
                                                                              rank_estado))
    st.write("Dentro do partido {}, seria o {}o.".format(partido,rank_partido))
    df_part_g=df_part_g[df_part_g["SG_UE"]==uf].copy()
    df_part_g.sort_values("ORGAO_GESTOR", ascending=True, inplace=True)
    #df_prop.rename(columns={"TOTAL_PROPOSTAS":"Total de propostas",
    #                        "NM_PUBLICO":"Nome"}, inplace=True)
    data = df_part_g
    fig =px.treemap(
        data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values="ORGAO_GESTOR")
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_participacao(cpf):
    st.header("Participações em orgãos ou comissões da Câmara")
    df_part=df_parlamentares[["CPF","NM_PUBLICO","SG_UE","SG_PARTIDO","ORGAO_PARTICIPANTE"]].copy()
    df_part.fillna(0, inplace=True)
    df_part["BR"]=df_part["ORGAO_PARTICIPANTE"].rank(ascending=False)
    df_part["UF"]=df_part.groupby("SG_UE")["ORGAO_PARTICIPANTE"].rank(ascending=False)
    df_part["PARTIDO"]=df_part.groupby("SG_PARTIDO")["ORGAO_PARTICIPANTE"].rank(ascending=False)
    parlamentar=df_part[df_part["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    part=parlamentar["ORGAO_PARTICIPANTE"].iloc[0]
    part=locale.str(part)
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])
    st.write("Com {} participações em orgãos ou comissões da câmara, é o {}o com mais participações, no {} seria o {}o".format(part,
                                                                              rank_br,
                                                                              uf,
                                                                              rank_estado))
    st.write("Dentro do partido {}, seria o {}o.".format(partido,rank_partido))
    df_part=df_part[df_part["SG_UE"]==uf].copy()
    df_part.sort_values("ORGAO_PARTICIPANTE", ascending=True, inplace=True)
    #df_prop.rename(columns={"TOTAL_PROPOSTAS":"Total de propostas",
    #                        "NM_PUBLICO":"Nome"}, inplace=True)
    data = df_part
    fig =px.treemap(
        data,path=['SG_UE', 'SG_PARTIDO',"NM_PUBLICO"], values="ORGAO_PARTICIPANTE")
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_gastos_gabinete(cpf):
    st.header("Gastos de gabinete - Acumulado")
    df_gastos=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","GASTO_GABINETE"]].copy()
    df_gastos.fillna(0, inplace=True)
    df_gastos["BR"]=df_gastos["GASTO_GABINETE"].rank(ascending=False)
    df_gastos["UF"]=df_gastos.groupby("SG_UE")["GASTO_GABINETE"].rank(ascending=False)
    df_gastos["PARTIDO"]=df_gastos.groupby("SG_PARTIDO")["GASTO_GABINETE"].rank(ascending=False)
    parlamentar=df_gastos[df_gastos["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    uf=parlamentar["SG_UE"].iloc[0]
    gastos=parlamentar["GASTO_GABINETE"].iloc[0]
    loc=locale.setlocale( locale.LC_ALL, '' )
    gastos=locale.currency(gastos, grouping=True )
    rank_br=int(parlamentar["BR"].iloc[0])
    rank_estado=int(parlamentar["UF"].iloc[0])
    rank_partido=int(parlamentar["PARTIDO"].iloc[0])
    st.write("Com gastos de gabinete acumulados de {}, no Brasil é o {}o. colocado em gastos de gabinete, no estado {} é o {}o".format(gastos,
                                                                                                  rank_br,
                                                                                                  uf,
                                                                                                  rank_estado))
    st.write("Dentro do partido {}, foi o {}o.".format(partido, rank_partido))
    df_gastos=df_gastos[df_gastos["SG_UE"]==uf].copy()
    data = df_gastos
    fig =px.treemap(
          data,path=['SG_UE', 'SG_PARTIDO',"NM_CANDIDATO"], values='GASTO_GABINETE')
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_gastos_evolucao(cpf):
    st.header("Gastos de gabinete - Evolução Mensal")
    df_cpf_gastos=df_gastos_gabinete[df_gastos_gabinete["CPF"]==cpf]
    df_cpf_gastos.fillna(0,inplace=True)
    df_cpf_gastos=df_cpf_gastos.rename(columns={"VL_GASTO_BRUTO":"Valor",
                                                "DT_GASTO":"Mês"})
    df_cpf_gastos.index=df_cpf_gastos["Mês"]
    gastos=pd.DataFrame(df_cpf_gastos.resample("M")["Valor"].sum())
    tipos=list(df_cpf_gastos["DS_GASTO"].unique())
    for tipo in tipos:
        gastos[tipo]=df_cpf_gastos[df_cpf_gastos["DS_GASTO"]==tipo].resample("M")["Valor"].sum()
    gastos.fillna(0, inplace=True)
    gastos.drop(columns="Valor", inplace=True)
    gastos.head()
    data=gastos
    fig = px.bar(data, height=400)
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_votacoes(cpf):
    st.header("Votações")
    #
    #
    return




lista=list(df_parlamentares["SG_UE"].unique())
lista_uf=sorted(lista)
uf=lista_uf[0]

st.title("Parlamentar")
#st.sidebar.write("Selecione o estado")

uf = st.sidebar.selectbox("Selecione o Estado",lista_uf)
#st.write("Estado {}".format(uf))

lista=list(df_parlamentares[df_parlamentares["SG_UE"]==uf]["NM_PUBLICO"])
lista_parlamentar=sorted(lista)
nome=lista_parlamentar[0]

nome = st.sidebar.selectbox("selecione o parlamentar",lista_parlamentar)
#st.write("Parlamentar {}".format(nome))


cpf=df_original[df_original["NM_PUBLICO"]==nome]["CPF"].iloc[0]
id=df_original[df_original["NM_PUBLICO"]==nome]["ID_CAMARA"].iloc[0]
id=str(id)
#st.write(id)
foto="https://www.camara.leg.br/internet/deputado/bandep/"+id+".jpg"
st.sidebar.image(foto)

parlamentar=df_original[df_original["CPF"]==cpf]
cargo=parlamentar["DS_CARGO"].iloc[0]
df_parlamentares=df_original[df_original["DS_CARGO"]==cargo]

df_cpf_votacao=df_municipios_votacao[df_municipios_votacao["CPF"]==cpf]
df_cpf=df_parlamentares[df_parlamentares["CPF"]==cpf]

exibe_cadastro(cpf)
exibe_gestao(cpf)
exibe_participacao(cpf)
exibe_propostas(cpf)
exibe_assiduidade(cpf)
exibe_gastos_gabinete(cpf)
exibe_gastos_evolucao(cpf)
exibe_gastos_campanha(cpf)
exibe_votacao(cpf)
exibe_votacao_estado(df_cpf, df_cpf_votacao)
exibe_twitter(cpf)
exibe_bens(cpf)
exibe_genero(cpf)
exibe_cor(cpf)
exibe_estado_civil(cpf)
exibe_instrucao(cpf)
exibe_ocupacao(cpf)


