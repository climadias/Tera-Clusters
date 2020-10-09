#
#Bibliotecas básicas
#
import pandas as pd
import numpy as np
from numpy import unique
from numpy import where
import datetime
import sqlite3 as sql

#
#Bibliotecas visualização
#
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 8))
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import streamlit as st

#
#Bibliotecas ML
#
from sklearn.cluster import KMeans
#nomalizing data to its std (x_new= x/std(x))
from scipy.cluster.vq import whiten


#
# carrega tabela para análise de base sql
#
con = sql.connect("../../dados/sql/base_completa.db")
df=pd.read_sql("select * from tabela_analise",con)
con.close()

df_original=df.copy()

variaveis_numericas=list(df.select_dtypes(include=[np.number]).columns)
variaveis_categoricas= list(df.select_dtypes(include="category").columns)
variaveis=variaveis_numericas+variaveis_categoricas
variaveis_modelo=variaveis

st.title("Clusterização")
#
# Carrega as variáveis contínuas já selecionadas inicialmente
#
continuas=['VL_BENS', 'VR_DESPESA_CONTRATADA', 'followers_count', 
                 'tweets', 'PERC_VOTOS_PARL_EST','ORGAO_PARTICIPANTE',
                 "ORGAO_TOTAL", 'ORGAO_GESTOR', 'PERC_PRESENCA', 'TOTAL_PROPOSTAS',
                 'GASTO_GABINETE']
basicas=continuas

# 
# Chamada streamlit sidebar
#
variaveis_modelo = st.sidebar.multiselect("Selecione as variáveis",variaveis,default=basicas)

# função para visualizar a escala das variáveis. Não utilizada no streamlit
def verifica_escala(preditores,df):
    x=df[preditores].values
    plt.plot(x.min(axis=0),"o", label="min")
    plt.plot(x.max(axis=0),"^", label="max")
    plt.legend(loc="best")
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    print(df[preditores].max().round(3))
    st.pyplot()
    

# Chamada para visualizar escala das variáveis. Não chamada no streamlit    
#verifica_escala(variaveis_modelo,df)

# Utilização do MinMax para ajuste da escala da variáveis
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(df[variaveis_modelo].values)
x=scaler.transform(df[variaveis_modelo].values)
df_scaled=pd.DataFrame(x,columns=variaveis_modelo)
df=df_scaled.copy()

# Visualizando com PCA

st.title("Visualização PCA")

x=df.values
variaveis=list(df.columns)

from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=3, random_state=42)
# fit PCA model 
pca.fit(x)
# transform data onto the first two principal components
x_pca = pca.transform(x)
#print("Original shape: {}".format(str(x.shape)))
#print("Reduced shape: {}".format(str(x_pca.shape)))
df_pca=pd.DataFrame(x_pca,columns=["PC-1","PC-2","PC-3"])


plt.figure(figsize=(8, 8))
plt.scatter(x_pca[:, 0], x_pca[:, 1])
#plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
st.pyplot()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca['PC-1'], df_pca['PC-2'], df_pca['PC-3'], c='skyblue', s=30)
ax.set_xlabel("First principal component")
ax.set_ylabel("Second principal component")
ax.set_zlabel("third principal component")
ax.view_init(20, -120)
st.pyplot()

df_pca_componentes=pd.DataFrame(pca.components_, columns=variaveis)
print("PCA component shape: {}".format(df_pca_componentes.shape))
#df_pca_componentes
plt.figure(figsize=(16, 4))
sns.heatmap(df_pca_componentes.T,annot=True, cmap="RdBu")
#(correlação,annot=True, vmin=-1, vmax=1)
st.pyplot()

def plot_pca_2d(x,y=True):
    plt.figure(figsize=(8, 8))
    if y.all():
        plt.scatter(x[:, 0], x[:, 1])
        plt.gca().set_aspect("equal")
    else:
        clusters = unique(y)
        for cluster in clusters:
            row_ix = where(y == cluster)
            # create scatter of these samples
            plt.scatter(x[row_ix, 0], x[row_ix, 1])
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.legend(clusters, loc="best")
    plt.show()
    if clusters[0]==-1:
        quant=np.bincount(y+1)
    else:
        quant=np.bincount(y)
    frame=pd.DataFrame({"cluster":clusters,"quant":quant})
    st.text("Número de componentes \n{}".format(frame))
    st.pyplot()
    
def plot_pca_3d(x,y,elevacao,azimute):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if y.all():
        ax.scatter(x[:, 0], x[:, 1],x[:,2])
        ax.gca().set_aspect("equal")
    else:
        clusters = unique(y)
        for cluster in clusters:
            row_ix = where(y == cluster)
            # create scatter of these samples
            ax.scatter(x[row_ix, 0], x[row_ix, 1],x[row_ix, 2], s=20)
    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")
    ax.set_zlabel("PC-3")
    ax.legend(clusters, loc='upper left')
    ax.view_init(elevacao, azimute)
    plt.show()
    if clusters[0]==-1:
        quant=np.bincount(y+1)
    else:
        quant=np.bincount(y)
    frame=pd.DataFrame({"cluster":clusters,"quant":quant})
    st.write("Número de componentes \n{}".format(frame))
    st.pyplot()
    
def plot_pca_2d_px(df,cluster):
    df["cluster"]=cluster
    fig=px.scatter(df, x='PC-1', y='PC-2', color=cluster, opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
    if clusters[0]==-1:
        quant=np.bincount(cluster+1)
    else:
        quant=np.bincount(cluster)
    frame=pd.DataFrame({"cluster":clusters,"quant":quant})
    st.write("Número de componentes \n{}".format(frame))
    
    
def plot_pca_3d_px(df,cluster):
    df["cluster"]=cluster
    fig=px.scatter_3d(df_pca, x='PC-1', y='PC-2', z='PC-3', color=cluster, opacity=0.7, width=1200, height=800)
    st.plotly_chart(fig, use_container_width=True)
    if clusters[0]==-1:
        quant=np.bincount(cluster+1)
    else:
        quant=np.bincount(cluster)
    #frame=pd.DataFrame({"cluster":clusters,"quant":quant})
    #st.write("Número de componentes \n{}".format(frame))
    
    
def plot_componentes(df):
    plt.figure(figsize=(16, 4))
    sns.heatmap(df.T,annot=True, cmap="RdBu")
    st.pyplot()

elevacao=st.sidebar.slider("Elevação PCA-3d", min_value=1, max_value=90, value=20, step=1, format=None, key="elevacao")
azimute=st.sidebar.slider("Azimute PCA -3d", min_value=-180, max_value=180, value=-120, step=5, format=None, key="azimute")
    
    
# Rodando algorítimos
x=df.values

# K-Means
#Verificando o número de clusters via inércia
st.title("Verificando o número de clusters via inércia")

kvalues=range(1,50)
kvalues
inercia=[]
st.title("Gráfico de cotovelo")

for k in kvalues:
    modelo=KMeans(n_clusters=k, init='k-means++',random_state=42)
    modelo.fit(x)
    inercia.append(modelo.inertia_)
df_inercia=pd.DataFrame({"Inercia":inercia, "K":kvalues}) 
df_inercia.plot("K","Inercia", marker='o')
st.pyplot()

k=st.sidebar.slider("Número de clusters (k-means, GMM, AHC)", min_value=1, max_value=50, value=10, step=1, format=None, key="knn")

# defining the kmeans function with initialization as k-means++
modelo = KMeans(n_clusters=k, init='k-means++')

# fitting the k means algorithm on scaled data
modelo.fit(x)
yhat = modelo.predict(x)
clusters = unique(yhat)
#print(variaveis_modelo)

st.header("KMeans")

plot_pca_2d(x_pca,yhat)
plot_pca_3d(x_pca,yhat, elevacao,azimute)
plot_pca_2d_px(df_pca,yhat)
plot_pca_3d_px(df_pca,yhat)
plot_componentes(df_pca_componentes)

df_kmeans=df_original.copy()
df_kmeans["cluster"]=yhat

df_kmeans.to_csv("./kmeans.csv", sep=";", index=False)

# Mean-shift

st.header("Mean-Shift")

from sklearn.cluster import MeanShift
modelo = MeanShift()
# fitting the k means algorithm on scaled data
modelo.fit(x)
yhat = modelo.predict(x)
clusters = unique(yhat)
#print(preditores)
#clusters


plot_pca_2d(x_pca,yhat)
plot_pca_3d(x_pca,yhat,elevacao,azimute)
#plot_pca_3d_px(df_pca,yhat)
plot_componentes(df_pca_componentes)

df_mean_s=df_original.copy()
df_mean_s["cluster"]=yhat

df_mean_s.to_csv("./mean_s.csv", sep=";", index=False)

## DBSCAN
st.header("DBSCAN")

eps=st.sidebar.slider("DBSCAN - eps", min_value=0.01, max_value=15.0, value=0.5, step=0.05, format=None, key="DB_eps")

min_part=st.sidebar.slider("DBSCAN - mínimo de participantes", min_value=1, max_value=40, value=3, step=1, format=None, key="DB_part")

from sklearn.cluster import DBSCAN
modelo = DBSCAN(eps=eps, min_samples=min_part)
modelo.fit(x)
yhat = modelo.fit_predict(x)
clusters = unique(yhat)
#print(preditores)
#clusters

plot_pca_2d(x_pca,yhat)
plot_pca_3d(x_pca,yhat,elevacao,azimute)
plot_pca_3d_px(df_pca,yhat)

plot_componentes(df_pca_componentes)

df_dbscan=df_original.copy()
df_dbscan["cluster"]=yhat

df_dbscan.to_csv("./dbscan.csv", sep=";", index=False)

# GMM
st.header("GaussianMixture")

from sklearn.mixture import GaussianMixture
modelo = GaussianMixture(n_components=k,random_state=42)
modelo.fit(x)
yhat = modelo.predict(x)

plot_pca_2d(x_pca,yhat)
plot_pca_3d(x_pca,yhat,elevacao,azimute)
plot_pca_3d_px(df_pca,yhat)

plot_componentes(df_pca_componentes)

df_GMM=df_original.copy()
df_GMM["cluster"]=yhat

df_GMM.to_csv("./GMM.csv", sep=";", index=False)

# Aglomerative H Cluste
st.header("AgglomerativeClustering")

from sklearn.cluster import AgglomerativeClustering
modelo = AgglomerativeClustering(n_clusters=k)
yhat = modelo.fit_predict(x)
clusters = unique(yhat)

plot_pca_2d(x_pca,yhat)
plot_pca_3d(x_pca,yhat,elevacao,azimute)
plot_pca_3d_px(df_pca,yhat)

plot_componentes(df_pca_componentes)

df_HC=df.copy()
df_HC["cluster"]=yhat

df_HC.to_csv("./HC.csv", sep=";", index=False)