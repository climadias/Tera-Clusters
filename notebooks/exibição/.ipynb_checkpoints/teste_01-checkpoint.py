import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns


#Dados de parlamentares
df_parlamentares=pd.read_csv("../dados/detalhes_parlamentares.csv", 
                             encoding="ISO-8859-1",
                             header=0)
df_parlamentares.drop(["DS_SIT_TOT_TURNO", "ST_REELEICAO","ST_DECLARAR_BENS",
                       "CD_TIPO_BEM_CANDIDATO", "DS_TIPO_BEM_CANDIDATO", 
                       "DS_BEM_CANDIDATO","VR_BEM_CANDIDATO"], axis=1, inplace=True)
df_parlamentares=df_parlamentares.drop_duplicates("NR_CPF_CANDIDATO",keep="first")
lista_partidos=list(df_parlamentares["SG_PARTIDO"].unique())
lista_partidos.sort()

rj=gpd.read_file("../dados/33MUE250GC_SIR.shp")

st.title("Partidos")
#st.sidebar.write("Selecione o partido")

partido = st.sidebar.selectbox("selecione o partido",lista_partidos)
st.write("Partido escolhido {}".format(partido))

partido_parlamentares=df_parlamentares[df_parlamentares["SG_PARTIDO"]==partido]
cargos=partido_parlamentares["DS_CARGO"].value_counts()

st.bar_chart(cargos)
fig, ax = plt.subplots(figsize=(12,8))
rj.plot(ax=ax,  cmap="Blues", edgecolor="black", linewidth=0.2)
ax.axis('off')
plt.axis("equal")
plt.show()

st.pyplot()


