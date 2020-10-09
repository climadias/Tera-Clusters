#
#Bibliotecas básicas
#
import pandas as pd
import numpy as np
import sqlite3 as sql

#
#Bibliotecas visualização
#
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 8))
import seaborn as sns
import streamlit as st

#
#Bibliotecas ML
#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble, tree
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

#
# carrega tabela para análise de base sql
#
con = sql.connect("../../dados/sql/base_completa.db")
df=pd.read_sql("select * from tabela_classificação",con)
con.close()

variaveis=list(df.columns)
variaveis.remove("CPF")
target=["cluster"]

preditores=list(set(variaveis)-set(target))
variaveis=target+preditores
correlação=df[variaveis].corr()


st.title("Correlações")

plt.rc('figure', figsize=(12, 8))
sns.heatmap(correlação,annot=True, vmin=-1, vmax=1, cmap="RdBu")
st.pyplot()

#plt.rc('figure', figsize=(20, 16))
sns.pairplot(df, y_vars=target, x_vars=preditores, hue=target[0])
st.pyplot()


#
# Colocando na mesma escala
#
def verifica_escala(preditores,df):
    x=df[preditores].values
    #plt.rc('figure', figsize=(12, 8))
    plt.plot(x.min(axis=0),"o", label="min")
    plt.plot(x.max(axis=0),"^", label="max")
    plt.legend(loc="best")
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    st.write(df[preditores].max().round(3))
    st.pyplot()


verifica_escala(preditores, df)

scaler = MinMaxScaler()
scaler.fit(df[preditores].values)
x=scaler.transform(df[preditores].values)
df_scaled=pd.DataFrame(x,columns=preditores)
df_scaled[target]=df[target]
df_scaled.head()
df_original=df.copy()
df=df_scaled.copy()

#
## treino e teste
#
df_treino, df_teste= train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])
X_treino=df_treino[preditores].values
y_treino=(df_treino[target].values).ravel()
X_treino.shape
y_treino.shape
X_teste=df_teste[preditores].values
y_teste=np.array(df_teste[target]).ravel()

#
# Função que executa algorítimo e registra valores
#
def classificação(modelo,target,preditores,treino,teste):
    x_treino=treino[preditores].values
    y_treino=treino[target].values
    y_treino=y_treino.ravel()
    
    x_teste=teste[preditores].values
    y_teste=teste[target].values
    
    modelo.fit(x_treino, y_treino)
    y_treino_pred=modelo.predict(x_treino)
    y_teste_pred=modelo.predict(x_teste)
    st.write(modelo)
    plot_confusion_matrix(modelo, x_teste,y_teste, cmap='Blues')
    st.pyplot()
    st.write("Acurácia no treino: {:.3f}".format(modelo.score(x_treino, y_treino)))
    st.write("Acuracia no teste: {:.3f}".format(modelo.score(x_teste, y_teste)))
    st.write("Precisão no teste: {:.3f}".format(precision_score(y_teste, y_teste_pred,average="weighted")))
    st.write("Recall no teste: {:.3f}".format(recall_score(y_teste, y_teste_pred,average="weighted")))
    st.write("F1 no teste: {:.3f}".format(f1_score(y_teste, y_teste_pred,average="weighted")))
    #plot_roc_curve(modelo, x_teste, y_teste)
    mod={}
    mod["modelo"]=modelo
    mod["acuracia_teste"]=modelo.score(x_teste, y_teste)
    mod["acuracia_treino"]=modelo.score(x_treino, y_treino)
    return mod

lista_modelos=[]


## KNN - verificando o valor de K ideal
st.title("KNN")
st.sidebar.header("KNN")
x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=range(1,50)

ac_treino=[]
ac_teste=[]

st.header("Parâmetro K para melhor acurácia em teste")
for val in valores:
    modelo= KNeighborsClassifier(n_neighbors=val)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste", inplace=True, ascending=False)
#st.write(df_acuracia.head())
k=int(df_acuracia["Parametro"].iloc[0])

k=st.sidebar.slider("Parâmetro K", min_value=1, max_value=50, value=k, step=1, format=None, key="knn-k")

#
# Executando Knn com o parâmetro ótimo
#
modelo = KNeighborsClassifier(n_neighbors=12)
st.header("Resultado")
result=classificação(modelo,target, preditores, df_treino, df_teste)
modelo=result["modelo"]
lista_modelos.append(result)

#
## Árvore de Decisão 
#
st.title("Árvore de decisão")
st.sidebar.header("Árvore de decisão")
st.header("Parâmetro max_depth para melhor acurácia em teste")

x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=range(2,50)

ac_treino=[]
ac_teste=[]
    
for val in valores:
    modelo= DecisionTreeClassifier(max_depth=val, random_state=42)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste", inplace=True, ascending=False)
st.write(df_acuracia.head(50))
max_d=int(df_acuracia["Parametro"].iloc[0])

max_d=st.sidebar.slider("Parâmetro max_depth", min_value=1, max_value=50, value=max_d, step=1, format=None, key="ad")

modelo = DecisionTreeClassifier(max_depth=max_d)

result=classificação(modelo,target, preditores, df_treino, df_teste)
modelo=result["modelo"]
lista_modelos.append(result)


st.header("Visualização da árvore")

#plt.figure(figsize=(12,8))
plot_tree(modelo, feature_names=preditores, filled=True,
          class_names=["0","1","2","3","4"]);

st.pyplot()

#
# Feature importance
#
st.header("Relevância das variáveis")

#modelo.feature_importances_
importancia=pd.Series(modelo.feature_importances_, index=preditores)
importancia=importancia.sort_values(ascending=True)
importancia.plot(kind="barh")
st.pyplot()

#
## SVC 
#
st.title("Support Vector Machine")
st.sidebar.header("Support Vector Machine")
st.header("Parâmetro C para melhor acurácia em teste")
gamma=0.2
gamma=st.sidebar.slider("Parâmetro gamma", min_value=0.01, max_value=5.0, value=gamma, step=0.01, format=None, key="gamma")

x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=range(1,50)

ac_treino=[]
ac_teste=[]
    
for val in valores:
    modelo=  SVC(C=val, gamma=gamma)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste", inplace=True, ascending=False)
st.write(df_acuracia.head())
c=int(df_acuracia["Parametro"].iloc[0])
c=st.sidebar.slider("Parâmetro C", min_value=1, max_value=50, value=c, step=1, format=None, key="c")


st.header("Parâmetro gamma para melhor acurácia em teste")
x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=list(np.arange(0.1,5,0.2))

ac_treino=[]
ac_teste=[]
    
for val in valores:
    modelo=  SVC(C=c, gamma=val)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste", inplace=True, ascending=False)
st.write(df_acuracia.head())


modelo = SVC(C=c, gamma=gamma)


result=classificação(modelo,target, preditores, df_treino, df_teste)
modelo=result["modelo"]
lista_modelos.append(result)


#
## Random Forest
#
st.title("Random Forest")
st.sidebar.header("Random Forest")
st.header("Parâmetro n_estimators para melhor acurácia em teste")

x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=list(np.arange(1,100,1))

ac_treino=[]
ac_teste=[]
    
for val in valores:
    modelo=  RandomForestClassifier(n_estimators=val, random_state=42)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste",inplace=True, ascending=False)
st.write(df_acuracia.head(40))

n_estim=int(df_acuracia["Parametro"].iloc[0])

n_estim=st.sidebar.slider("Parâmetro N_estimators", min_value=1, max_value=100, value=n_estim, step=1, format=None, key="n_estim")

modelo = RandomForestClassifier(n_estimators=n_estim, random_state=42)
result=classificação(modelo,target, preditores, df_treino, df_teste)
modelo=result["modelo"]
lista_modelos.append(result)

#
# Gradient Boosting
#
st.title("Gradient Boosting")
st.sidebar.header("Gradient Boosting")

st.header("Parâmetro learning_rate  para melhor acurácia em teste")

x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=list(np.arange(0.01,1,0.02))

ac_treino=[]
ac_teste=[]
    
for val in valores:
    modelo=  GradientBoostingClassifier(random_state=42,learning_rate=val)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste",inplace=True, ascending=False)
st.write(df_acuracia.head(40))

l_rate=float(df_acuracia["Parametro"].iloc[0])
l_rate=st.sidebar.slider("Parâmetro learning_rate", min_value=0.01, max_value=1.0, value=l_rate, step=0.01, format=None, key="l_rate")

modelo = GradientBoostingClassifier(random_state=42,learning_rate=l_rate)
result=classificação(modelo,target, preditores, df_treino, df_teste)
modelo=result["modelo"]
lista_modelos.append(result)

#
# XGBoosting
#
st.title("XGBoost")
st.sidebar.header("XGBoost")
st.header("Parâmetro learning_rate  para melhor acurácia em teste")

x_treino=df_treino[preditores].values
y_treino=df_treino[target].values
y_treino=y_treino.ravel()
x_teste=df_teste[preditores].values
y_teste=df_teste[target].values

valores=list(np.arange(0.01,1,0.01))

ac_treino=[]
ac_teste=[]
    
for val in valores:
    modelo=   xgb.XGBClassifier(random_state=42,learning_rate=val)
    modelo.fit(x_treino, y_treino)
    #y_treino_pred=modelo.predict(x_treino)
    #y_teste_pred=modelo.predict(x_teste)
    ac_treino.append(modelo.score(x_treino, y_treino))
    ac_teste.append(modelo.score(x_teste, y_teste))
    
df_acuracia=pd.DataFrame({"Parametro":valores,"acuracia_treino":ac_treino,"acuracia_teste":ac_teste }, index=valores) 
df_acuracia[["acuracia_treino","acuracia_teste"]].plot()
st.pyplot()
df_acuracia.sort_values("acuracia_teste",inplace=True, ascending=False)
st.write(df_acuracia.head(10))

l_rateX=float(df_acuracia["Parametro"].iloc[0])
l_rateX=st.sidebar.slider("Parâmetro learning_rate", min_value=0.01, max_value=1.0, value=l_rateX, step=0.01, format=None, key="l_rateX")

modelo = xgb.XGBClassifier(random_state=42, learning_rate=l_rateX)
result=classificação(modelo,target, preditores, df_treino, df_teste)
modelo=result["modelo"]
lista_modelos.append(result)

st.title("Comparação dos modelos")
modelos=pd.DataFrame(lista_modelos)
modelos.sort_values("acuracia_teste", ascending=False, inplace=True)
st.write(modelos[["modelo","acuracia_teste"]])