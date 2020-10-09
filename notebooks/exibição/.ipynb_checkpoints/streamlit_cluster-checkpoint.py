#Bibliotecas básicas
import pandas as pd
import numpy as np
from numpy import unique
from numpy import where

#Bibliotecas visualização
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Bibliotecas especificas
import streamlit as st
import datetime
import sqlite3
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from scipy.cluster.vq import whiten
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
import umap
#****************************************************************************************

#########################
#   INSTRUCOES GERAIS   #
#########################

#1) Salvar os arquivos no mesmo local deste script.py:
#	a) base_completa.db
#	b) kmean.csv
#	c) GMM.csv
#2) Verificar se as bibliotecas utilizadas estão instalados
#3) Verificar se o streamlit está instalado
#4) No prompt de comando verificar se a pasta atual é a mesma do script
#5) No prompt de comando digitar "streamlit run streamlit_cluster.py"



###############http://www.dominiopublico.gov.br/download/texto/ua000178.pdf
#   SUMARIO   #
###############

#1-Extracao dos dados
#2-Sidebar
#3-Cabecalho
#4-Grafico de Radar
#	.1-Preparacao dados + metodologia de notas dos clusters
#	.2-Grafico Radar
#-PCA
#-UMAP
#-Grafico Treemap

#****************************************************************************************
#DEFINICOES GERAIS
pd.options.display.float_format = "{:,.2f}".format
plt.rc('figure', figsize=(12, 8))

DADOS_BASICOS = 'base_completa.db'
DADOS_KMEANS = 'kmeans.csv'
DADOS_GMM = 'GMM.csv'
DESCONSIDERAR_NAO_EXERCICIO = True
ESCALA = MinMaxScaler()
ESCALA_RADAR_MAX = 10
SEED = 42
EIXOS = 3

#****************************************************************************************

####################################
#   PARTE 1 - EXTRACAO DOS DADOS   #
####################################

#Base 1: cadastro basico dos parlamentares
con = sqlite3.connect(DADOS_BASICOS)
df_basico=pd.read_sql("select * from kpi_cadastro_basico",con)
con.close()

#Base 2: clusters estimados por K-Means
df_kmeans = pd.read_csv(DADOS_KMEANS,sep=';',usecols=['cluster'],dtype={'cluster':int})
df_kmeans.rename(columns={'cluster':'kmeans'}, inplace=True)

#Base 3: clusters estimados por GMM
df_gmm = pd.read_csv(DADOS_GMM ,sep=';',usecols=['cluster'],dtype={'cluster':int})
df_kmeans.rename(columns={'cluster':'gmm'}, inplace=True)

#Juntando Base1 + Base2 + Base3 e ajustes
df_basico = pd.concat([df_basico,df_kmeans, df_gmm],axis =1)
df_basico['quantidade']=1
df_basico.fillna(0, inplace=True)
df_basico.drop_duplicates(inplace=True)

#condicional experimental: removendo os parlamentares não exercentes atualmente
if (DESCONSIDERAR_NAO_EXERCICIO == True):
    x = df_basico
    x = x[x['NM_CANDIDATO'] !='MARCELO HENRIQUE TEIXEIRA DIAS']
    x = x[x['NM_CANDIDATO'] !='LUIS ANTONIO FRANCISCATTO COVATTI']
    x = x[x['NM_CANDIDATO'] !='SANDRO ALEX CRUZ DE OLIVEIRA']
    x = x[x['NM_CANDIDATO'] !='RUBENS PEREIRA E SILVA JUNIOR']
    x = x[x['NM_CANDIDATO'] !='PAULO ROBERTO FOLETTO']
    x = x[x['NM_CANDIDATO'] !='LUIZ FLAVIO GOMES']
    x = x[x['NM_CANDIDATO'] !='JOSIAS GOMES DA SILVA']
    x = x[x['NM_CANDIDATO'] !='WAGNER MONTES DOS SANTOS']
    x = x[x['NM_CANDIDATO'] !='JEAN WYLLYS DE MATOS SANTOS']
    x = x[x['NM_CANDIDATO'] !='ONYX DORNELLES LORENZONI']
    x = x[x['NM_CANDIDATO'] !='TEREZA CRISTINA CORREA DA COSTA DIAS']
    df_basico = x


#########################
#   PARTE 2 - SIDEBAR   #
#########################

#PARTE 2.1 - ESCOLHA DO CLUSTER
#escolher qual cluster a ser visualizados
lista_cluster = ("Cluster 0",
                 "Cluster 1",
                 "Cluster 2",
                 "Cluster 3",
                 'Cluster 4')

qual_cluster = st.sidebar.selectbox(
    "Selecione o cluster",
    lista_cluster
)

dict_cluster = {
    lista_cluster[0]:0,
    lista_cluster[1]:1,
    lista_cluster[2]:2,
    lista_cluster[3]:3,        
    lista_cluster[4]:4
}

n_cluster = dict_cluster[qual_cluster]

###########################
#   PARTE 3 - CABECALHO   #
###########################

st.title('Hello world.')

st.markdown(
'O método científico refere-se a um aglomerado de regras básicas dos procedimentos que produzem o conhecimento científico, quer um novo conhecimento, quer uma correção (evolução) ou um aumento na área de incidência de conhecimentos anteriormente existentes.'
)

###############################
#   PARTE 3 - GRÁFICO RADAR   #
###############################

#PARTE 3.1 - DADOS
    
#separar a tabela
df = df_basico[['PERC_PRESENCA',
              'followers_count',
              'GASTO_GABINETE',
              'TOTAL_PROPOSTAS',
              'ORGAO_PARTICIPANTE',
              'ORGAO_GESTOR',
              'kmeans']]
df.sort_values(by='kmeans', inplace=True)  

#sera usado em seguida
df3=pd.DataFrame()

#aplicando um for poderoso que vai criar a tabela a ser aplicada no grafico de aranha
for i in df['kmeans'].unique():

    #separando em cluster
    df1 = df[df['kmeans']==i].drop(columns=['kmeans'])
    
    #aplicando a mediana em cada cluster
    ar1 = df1.median() 
    
    #transpondo o resultado para melhor leitura
    df2 = pd.DataFrame(ar1).T
    
    #juntando os grupos
    df3 = pd.concat([df3,df2],axis=0)
    
#aplicando a escala definida no inicio do script
ar2 = ESCALA.fit_transform(df3)*ESCALA_RADAR_MAX

#Alteracoes na metodologia de pontuacao dos clusteres devem ser aletrados no cabecalho.
#16/09: primeiramente as notas foram escalas entre 0 a 10 pelo metodo de minimos e
#maximos e depois segregados entre cluster.

#ultimos ajustes
df4 = pd.DataFrame(ar2).rename(columns={0:'Presença (%)',
                                       1:'Seguidores do Twitter',
                                       2:'Gastos de gabinete',
                                       3:'Produção parlamentar',
                                       4:'Quantidade de participações em órgãos',
                                       5:'Quantidade de órgãos geridos'})
df4.reset_index(drop=True, inplace=True)

#salvando o arquivo para ser usado no outro streamlit
df4.to_csv('var_escala0a10_kmeans.csv', sep=';')

#convertendo para np.array os parametros a serem utilizados no grafico
valores = np.array(df4.iloc[n_cluster,:])
nomes = np.array(df4.columns)

n_cluster = 2

#e, finalmente, o grafico
#ex.: cor = px.colors.qualitative.D3
def grafico_aranha(cluster,cor):
    
    valores = np.array(df4.iloc[cluster,:])
    nomes = np.array(df4.columns)

    st.header('Cluster {}'.format(cluster))
    fig = px.line_polar(df4,
                        r = valores,
                        theta = nomes,
                        line_close = True,
                        range_r = (0,10),
                        color_discrete_sequence = cor
    )
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

grafico_aranha(0, px.colors.qualitative.D3)
grafico_aranha(1, px.colors.qualitative.Plotly_r)
grafico_aranha(2, px.colors.qualitative.Alphabet)
grafico_aranha(3, px.colors.qualitative.Dark24_r)
grafico_aranha(4, px.colors.qualitative.Dark2)

#########################
#   PARTE X - PERSONA   #
#########################

#bases e ajustes
df = df_basico[['VL_BENS',
                'IDADE',
                'VR_DESPESA_CONTRATADA',
                'IDHM_ponderado',
                'DS_GRAU_INSTRUCAO',
                'kmeans']]
df.sort_values(by='kmeans', inplace=True)               

#separando em grupos
x = df[df['kmeans']==n_cluster]

#populacao do cluster
result = pd.Series([len(x)], name='Informações gerais').rename(index={0:'População'})

#mediana das var numericas
x_median = x.iloc[:,1:5].median().rename(index={'IDADE':'Idade',
                                                'IDHM_ponderado':'IDHM médio dos eleitores',
                                                'VL_BENS':'Patrimônio (R$)',
                                                'VR_DESPESA_CONTRATADA':'Gastos eleitorais (R$)'})
                                                
result = pd.concat([result,x_median],axis=0)
result = pd.DataFrame(result).rename(columns={0:'Dados gerais'})

st.write(result)
#ERRO! Por algum motivo o 'Patrimônio(R$)'(VL_BENS) não está sendo aparecendo. O mesmo aconteceu com a escolaridade.

st.markdown(
'Para muitos autores, o método científico nada mais é do que a lógica aplicada à ciência. Os métodos que fornecem as bases lógicas o conhecimento científico são: método indutivo, método dedutivo, método hipotético-dedutivo, método dialético e método fenomenológico.'
)

#########################################
#   PARTE X - VISUALIZACAO PCA E UMAP   #
#########################################

st.header('Visualização')

#PARTE X.1 - DADOS
#definindo os dados
df = df_basico[['ORGAO_PARTICIPANTE',
               'ORGAO_GESTOR',
               'TOTAL_PROPOSTAS',
               'PERC_PRESENCA',
               'GASTO_GABINETE',
               'followers_count'
              ]].fillna(0)

#aplicando ajuste de escala MinMax
df_sc = MinMaxScaler().fit_transform(df)

#reaplicando os nomes
df = pd.DataFrame(df_sc).rename({0:'ORGAO_PARTICIPANTE',
               	                 1:'ORGAO_GESTOR',
                	             2:'TOTAL_PROPOSTAS',
                   	             3:'PERC_PRESENCA',
                   	      	     4:'GASTO_GABINETE',
                   		         5:'followers_count'
                          	   },axis=1)

#PARTE X.2 - PCA

#setando parametros iniciais
pca = PCA(n_components=EIXOS, random_state=SEED)

#usar o fit ou fit_transform?
embedding = pca.fit_transform(df)
#embedding = pca.fit(df)

#unificando o df reduzido + clusters
df_emb = pd.DataFrame(embedding)
result_pca = pd.concat([df_emb,df_basico['kmeans']], axis=1)


#PARTE X.3 - UMAP

#definindo parâmetros do algoritmo
n_neighbors = 400       #(max=nrows-1) quanto maior mais preserva as características globais
min_dist= 1             #(max = 1) quanto menor, mais apertado fica cluster
n_components = EIXOS    #dimensões
metric = 'euclidean'    #tipo de distância (euclidiana é uma reta entre 2 pontos)
random_state= SEED

#definindo o redutor
reducer = umap.UMAP(n_neighbors = n_neighbors,
                    n_components=n_components,
                    min_dist= min_dist,
                    metric = metric,
                    random_state = random_state
                   )

#treinando o redutor de dimensionalidade do UMAP
embedding = reducer.fit_transform(df)
#embedding = reducer.fit(df)

#unificando o df reduzido + clusters
df_emb = pd.DataFrame(embedding)
result_umap = pd.concat([df_emb,df_basico['kmeans']], axis=1)


#PARTE X.3 - GRAFICOS

#plostando os graficos de PCA e UMAP
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"},{"type": "scene"}]],
    subplot_titles=("PCA", "UMAP")
)


fig.add_trace(go.Scatter3d(x=result_pca[0],
                           y=result_pca[1],
                           z=result_pca[2],
                           mode="markers",
                           marker=dict(
                               size = 4,
                               color=np.where((result_pca['kmeans'] == n_cluster), 'blue', 'grey'),
                               opacity = 1
					        )                           
						),row=1, col=1
)

fig.add_trace(go.Scatter3d(x=result_umap[0],
                           y=result_umap[1],
                           z=result_umap[2],
                           mode="markers",
                           marker=dict(
                               size = 4,
                               color=np.where((result_umap['kmeans'] == n_cluster), 'red', 'grey'),
                               opacity = 1
					        )                           
						),row=1, col=2
)

fig.update_layout(#margin=dict(l=0, r=0, b=0, t=0),
                  height=400,
                  width=700,
                  showlegend=False
                 )
st.plotly_chart(fig, use_container_width=False)


st.markdown(
'A metodologia científica se reforça no pensamento de René Descartes, que foi posteriormente desenvolvido empiricamente pelo físico inglês Isaac Newton. Descartes propôs chegar à verdade através da dúvida sistemática e da decomposição do problema em pequenas partes, características que definiram a base da pesquisa científica.[nota 1] Compreendendo-se os sistemas mais simples, gradualmente se incorporam mais e mais variáveis, em busca da descrição do todo.' 
)


####################################
#   PARTE 6 - TREEMAP DE PARTIDO   #
####################################

#PARTE 6.1 - DADOS
df = pd.DataFrame(df_basico, columns=['SG_PARTIDO','kmeans','quantidade'])
for i in df['kmeans'].unique():
	df['kmeans'][df['kmeans']==i] = 'Cluster '+str(i)


#PARTE 6.2 - GRAFICO TREEMAP
st.header("Composição dos clusters por partidos")
fig = px.treemap(df, path=['kmeans', 'SG_PARTIDO'], values='quantidade',
				hover_name = 'quantidade',
                width=800,
                height=500
)

st.plotly_chart(fig, use_container_width=False)

st.write('Na minha terra tem palmeira')
st.write('Onde canda o sabia')
st.write('SenA CosB')
st.write('SenB CosA')

st.markdown(
'Descrições de métodos são encontradas desde as civilizações antigas, como no Antigo Egito e na Grécia Antiga, mas só foi na sociedade árabe, há cerca de mil anos que as bases do que seria o método científico atual foram sendo construídas.'
)


st.title('ACABA AQUI')

'''
########################################
#   PARTE 1 - EXPLORANDO OS CLUSTERS   #
########################################

#PARTE 1.1 - AJUSTES PRE GRAFICOS
# carrega tabela para análise (eventualmente mudar o caminho)
con = sql.connect(DATA_DIR)
df=pd.read_sql("select * from tabela_analise",con)
con.close()

df_original=df.copy()
 
#definindo as listas no menu dropdown
variaveis_numericas=list(df.select_dtypes(include=[np.number]).columns)
variaveis_categoricas= list(df.select_dtypes(include="category").columns)
variaveis=variaveis_numericas+variaveis_categoricas
variaveis_modelo=variaveis

#variaveis sugeridas ao abrir o streamlit
continuas = ['VL_BENS', 'VR_DESPESA_CONTRATADA', 'PERC_VOTOS_PARL_EST',  #variaveis pre eleicao
             'followers_count', 'tweets',                               #variaveis midia social
             'ORGAO_PARTICIPANTE', "ORGAO_TOTAL", 'ORGAO_GESTOR',       #variaveis de composicao de comissoes e orgaos
             'PERC_PRESENCA', 'TOTAL_PROPOSTAS', 'GASTO_GABINETE']      #variaveis de produtividade parlamentar

basicas=continuas
variaveis_modelo = st.sidebar.multiselect("Selecione as variáveis", variaveis, default=basicas)

#verifica_escala(variaveis_modelo,df)
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

#importante: escolher o ajuste de escala a ser feito (minmaxscaler ou standardscaler)
scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(df[variaveis_modelo].values)
x=scaler.transform(df[variaveis_modelo].values)
df_scaled=pd.DataFrame(x,columns=variaveis_modelo)
df=df_scaled.copy()


#PARTE 1.2 - INICIO DA VISUALIZACAO

# Visualizando com PCA
st.title("Clusterização")
st.title("Vizualização PCA")

x=df.values
variaveis=list(df.columns)


# keep the first two principal components of the data
pca = PCA(n_components=EIXOS, random_state=SEED)
# fit PCA model 
pca.fit(x)
# transform data onto the first two principal components
x_pca = pca.transform(x)
df_pca=pd.DataFrame(x_pca,columns=["PC-1","PC-2","PC-3"])

#plotando o PCA
plt.figure(figsize=(8, 8))
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
st.pyplot()


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


#PARTE 1.3 - ALGORITMOS DE CLUSTERIZAÇÃO


#grafico de cotovelo (k-means): verificando o número de clusters via inércia
st.title("Verificando o número de clusters via inércia")

kvalues=range(1,50)
kvalues
inercia=[]
st.title("Gráfico de cotovelo")

for k in kvalues:
    modelo=KMeans(n_clusters=k, init='k-means++',random_state=SEED)
    modelo.fit(x)
    inercia.append(modelo.inertia_)
df_inercia=pd.DataFrame({"Inercia":inercia, "K":kvalues}) 
df_inercia.plot("K","Inercia", marker='o')
st.pyplot()

k=st.sidebar.slider("Número de clusters (K-Means, GMM e AHC)", min_value=1, max_value=50, value=5, step=1, format=None, key="knn")


# ALGORITMO 1: K-MEANS (cluster)

# defining the kmeans function with initialization as k-means++
modelo = KMeans(n_clusters=k, init='k-means++')

# fitting the k means algorithm on scaled data
modelo.fit(x)
yhat = modelo.predict(x)
clusters = unique(yhat)

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


modelo = GaussianMixture(n_components=k,random_state=SEED)
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
'''


