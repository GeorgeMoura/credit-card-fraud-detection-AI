# -*- coding: utf-8 -*-
"""
@authors: Gabriel Gonçalves, George Moura

Este código gera o Dendograma(Clusterização hieráquica) e a Matriz de Correlação dos dados da base.
"""

print("Importando bibliotecas...")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyodbc
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder


#função responsável pela correção dos dados nulos da base, para entrada do classificador
def data_correction(dataset_train):
	for inc in dataset_train.columns:

		if(dataset_train[inc].dtype == np.float64):
			dataset_train[inc] = dataset_train[inc].fillna(-1000.0)

		elif(dataset_train[inc].dtype == np.int64):
			dataset_train[inc] = dataset_train[inc].fillna(-1000)

		elif(dataset_train[inc].dtype == np.datetime64):
			dataset_train[inc] = dataset_train[inc].fillna(Timestamp("00000000"))

		elif(dataset_train[inc].dtypes.name == 'bool'):
			dataset_train[inc] = dataset_train[inc].astype(float)
			dataset_train[inc] = dataset_train[inc].fillna(0.5)

		elif(dataset_train[inc].dtype == np.object):

			heuristic_limit = len(dataset_train[inc])
			iterator = heuristic_limit

			for i in dataset_train[inc]:
	            
				if(iterator == 0):
					dataset_train[inc] = dataset_train[inc].astype(float)
					dataset_train[inc] = dataset_train[inc].fillna(0.5)
				if(i == True or i == False or i == None):
					iterator = iterator - 1
				else:
					dataset_train[inc] = dataset_train[inc].fillna("-1000")
					break
	return dataset_train

################################IMPORTANDO DADOS######################
print("Conectando a base...")
connection = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=255.255.255.255;DATABASE=master;Trusted_Connection=Yes')

#Montagem da query, selecionar emissor e quantos dias à serem ignorados a partir de hoje.
emissor = "88"
date = "-1"
query = "SELECT ..."

print("Carregando Dataframe...")
dataset_train = pd.read_sql(query,connection)

#Mapeamento da label de decisão, 1 para Fraude, e 0 para Não-Fraude
dataset_train['Resolucao'] = dataset_train['Resolucao'].map({'Fraude': 1, 'Nao Fraude': 0})

#corrigindo dados com NULL
print("Corrigindo colunas com dados nulos...")

dataset_train = data_correction(dataset_train)

#Transfere o dataframe para uma variável manipulável por Index
num_cols = len(dataset_train.columns)
x_train = dataset_train.iloc[:, :num_cols].values

################################ LABELS ################################

'''
	Gera hashs numéricas exclusivas para cada novo dado do dataframe,
	o algoritmo de classificação só observa dados numéricos.
'''
print("Codificando dados...")
labelencoder_x = LabelEncoder()

for x in range(len(x_train[0])):
	x_train[:, x] = labelencoder_x.fit_transform(x_train[:, x])


#Transposição da matriz categórica para clustering hierárquico de atributos
x_train = x_train.transpose()

print("\n\nDendograma\n")
#Clustering Hierárquico

dendogram = sch.dendrogram(sch.linkage(x_train, method = 'ward'))
plt.title("Dendograma")
plt.ylabel("Distancias")
plt.show()

#Contador da distribuição de classes
sns.countplot(dataset_train['Resolucao'], label = "Quantidade")
plt.show()

#Renomeando as colunas para o seu index, leva a uma visualização mais clara da matriz de correlação (Opcional, você consegue dar zoom na matriz)
i = 0
for inc in dataset_train.columns:
	dataset_train.rename(columns={inc: str(i)}, inplace=True)
	i = i + 1

print("\n\nMatriz Correlações\n")
#Matriz de Correlações
corr_mat=dataset_train.corr(method='pearson')
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(30,30))
sns.heatmap(corr_mat,vmax=1,square=True,cmap='cubehelix')
plt.show()



