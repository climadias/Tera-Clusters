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
con.close()

df_original=df_parlamentares.copy()

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
            "DS_GRAU_INSTRUCAO","DS_OCUPACAO","DS_ESTADO_CIVIL","DS_CARGO","SG_PARTIDO","ANO_ELEICAO"]
    #parlamentar=df_parlamentar[campos].to_dict(orient="records")[0]
    parlamentar=df_parlamentar[campos]
    parlamentar.rename({"NM_CANDIDATO": "Nome","NM_URNA_CANDIDATO":"Nome na Urna",
                       "DT_NASCIMENTO":"Nascimento","DS_COR_RACA":"Cor",
                        "DS_GENERO":"Gênero","DS_ESTADO_CIVIL":"Estado Civil",
                        "DS_OCUPACAO":"Ocupação ant.","SG_PARTIDO":"Partido",
                        "DS_CARGO":"Cargo","ANO_ELEICAO":"Eleito em",
                       "DS_GRAU_INSTRUCAO":"Grau de instrução"}, inplace=True, axis=1)
    #parlamentar["BENS"]=valor_bens.round(2)
    st.write(parlamentar.T)
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
    df_genero=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","DS_GENERO"]].copy()
    df_genero=df_genero[df_genero["SG_UE"]==uf].copy()
    data = df_genero
    fig =px.sunburst(data,
                path=['SG_UE', "DS_GENERO","NM_CANDIDATO"])
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
    df_cor=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","DS_COR_RACA"]].copy()
    df_cor=df_cor[df_cor["SG_UE"]==uf].copy()
    data = df_cor
    fig =px.sunburst(data,
                path=['SG_UE', "DS_COR_RACA","NM_CANDIDATO"])
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
    df_civil=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","DS_ESTADO_CIVIL"]].copy()
    df_civil=df_civil[df_civil["SG_UE"]==uf].copy()
    data = df_civil
    fig =px.sunburst(data,
                path=['SG_UE',"DS_ESTADO_CIVIL","NM_CANDIDATO"])
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
    df_instruc=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","DS_GRAU_INSTRUCAO"]].copy()
    df_instruc=df_instruc[df_instruc["SG_UE"]==uf].copy()
    data = df_instruc
    fig =px.sunburst(data,
                path=['SG_UE',"DS_GRAU_INSTRUCAO","NM_CANDIDATO"])
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
    df_ocupacao=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","DS_OCUPACAO"]].copy()
    df_ocupacao=df_ocupacao[df_ocupacao["SG_UE"]==uf].copy()
    data = df_ocupacao
    fig =px.sunburst(data,
                path=['SG_UE',"DS_OCUPACAO","NM_CANDIDATO"])
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return


def exibe_bens(cpf):
    st.header("Bens do parlamentar")
    loc=locale.setlocale( locale.LC_ALL, '' )
    df_bens=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","VL_BENS"]].copy()
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
    data,path=['SG_UE', 'SG_PARTIDO',"NM_CANDIDATO"], values='VL_BENS')
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
    #
    #
    return

def exibe_twitter(cpf):
    st.header("Seguidores Twitter")
    df_twitter=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","followers_count"]].copy()
    df_twitter.fillna(0, inplace=True)
    df_twitter["FOLL_BR"]=df_twitter["followers_count"].rank(ascending=False)
    df_twitter["FOLL_UF"]=df_twitter.groupby("SG_UE")["followers_count"].rank(ascending=False)
    df_twitter["FOLL_PARTIDO"]=df_twitter.groupby("SG_PARTIDO")["followers_count"].rank(ascending=False)
    parlamentar=df_twitter[df_twitter["CPF"]==cpf]
    partido=parlamentar["SG_PARTIDO"].iloc[0]
    followers=parlamentar["followers_count"].iloc[0]
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
        data,path=['SG_UE', 'SG_PARTIDO',"NM_CANDIDATO"], values='followers_count')
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_gastos_campanha(cpf):
    st.header("Gastos de campanha")
    df_gastos=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","VR_DESPESA_CONTRATADA"]].copy()
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
        data,path=['SG_UE', 'SG_PARTIDO',"NM_CANDIDATO"], values='VR_DESPESA_CONTRATADA')
    st.plotly_chart(fig, use_container_width=True)
    return

def exibe_votacao(cpf):
    st.header("Votação")
    df_votos=df_parlamentares[["CPF","NM_CANDIDATO","SG_UE","SG_PARTIDO","VOTOS_TOTAL_PARL"]].copy()
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
        data,path=['SG_UE', 'SG_PARTIDO',"NM_CANDIDATO"], values='VOTOS_TOTAL_PARL')
    st.plotly_chart(fig, use_container_width=True)
    return
    #
    

def exibe_assiduidade(cpf):
    st.header("Assiduidade")
    #
    #
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
st.write("Estado {}".format(uf))

lista=list(df_parlamentares[df_parlamentares["SG_UE"]==uf]["NM_CANDIDATO"])
lista_parlamentar=sorted(lista)
nome=lista_parlamentar[0]

nome = st.sidebar.selectbox("selecione o parlamentar",lista_parlamentar)
st.write("Parlamentar {}".format(nome))

cpf=df_original[df_original["NM_CANDIDATO"]==nome]["CPF"].iloc[0]
parlamentar=df_original[df_original["CPF"]==cpf]
cargo=parlamentar["DS_CARGO"].iloc[0]
df_parlamentares=df_original[df_original["DS_CARGO"]==cargo]

df_cpf_votacao=df_municipios_votacao[df_municipios_votacao["CPF"]==cpf]
df_cpf=df_parlamentares[df_parlamentares["CPF"]==cpf]

exibe_cadastro(cpf)
exibe_genero(cpf)
exibe_cor(cpf)
exibe_estado_civil(cpf)
exibe_instrucao(cpf)
exibe_ocupacao(cpf)
exibe_bens(cpf)
exibe_twitter(cpf)
exibe_gastos_campanha(cpf)
exibe_votacao(cpf)
exibe_votacao_estado(df_cpf, df_cpf_votacao)
exibe_assiduidade(cpf)
exibe_votacoes(cpf)

