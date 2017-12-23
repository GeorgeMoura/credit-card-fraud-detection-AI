# -*- coding: utf-8 -*-
"""
@authors: Gabriel Gonçalves, George Moura

Este código contém o processo de busca por parâmetros ótimos dos 3 algoritmos para uma dada base. Consiste em um processamento pesado,
pois além dos testes de todas as combinações do conjunto de parâmetros para cada um dos 3 algorítmos, é feito um Cross-validation para
validar o resultado.
"""
print("Importando bibliotecas...")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.externals import joblib
import pyodbc
import json

def Classification_model_gridsearchCV(name, model, param_grid, data_X, data_y):
	f = open( name + "_grid_search_results.txt", "w" )
	start_time = time.time()

	# busca pela combinação de parâmetros que possui maior acurácia utilizando cross-validation
	clf = GridSearchCV(model,param_grid,cv=5,scoring="accuracy")

	# ajuste do modelo com os parâmetros encontrados
	clf.fit(data_X,data_y)

	# retorna parâmetros em porcentagem
	method_score = clf.best_score_*100

	processing_time = time.time() - start_time
	print("Melhor combinação de parâmetros encontrada:")
	print(clf.best_params_)
	f.write(json.dumps(clf.best_params_))
	print("Cross-validation score proporcionada por essa combinação de parâmetros: %s" %"{0:.3f}%".format(method_score))
	print("Processing time: %s seconds" % (processing_time))
	f.close()

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


################################ IMPORTANDO DADOS ######################

print("Conectando a base...")
connection = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=255.255.255.255;DATABASE=master;Trusted_Connection=Yes')

'''
	Montagem da query, selecionar emissor e o range de dias à serem consultados, esta abordagem viabiliza testes, pois 
	o processamento do grid-search é pesado.
'''
emissor = "88"
date0 = "-1"
date1 = "-2"
query = "SELECT co.DataCompra, co.Valor, co.ValorDestino, co.ValorOrigem, co.Pais, co.Origem, co.CodigoRespostaFraude, co.MCC, co.CodigoMoedaOrigem, co.CodigoMoedaDestino, co.ModoEntradaTerminal, co.Id_Estado, co.ID_Produto, co.DataSaidaCompra, co.FraudScore, tr.Descricao, ct.UF, ct.LimiteGlobal, ct.LimiteGlobalCliente, ct.LimiteSaqueNacGlobal, ct.QtdComprasDiaAprovadas, ct.QtdCompras3DiasAprovadas, ct.QtdCompras7DiasAprovadas, ct.QtdComprasTotalAprovadas, ct.QtdComprasDiaNegadas, ct.QtdCompras3DiasNegadas, ct.QtdCompras7DiasNegadas, ct.QtdComprasTotalNegadas, ct.ValorComprasDiaAprovadas, ct.ValorCompras3DiasAprovadas, ct.ValorCompras7DiasAprovadas, ct.ValorComprasDiaNegadas, ct.ValorCompras3DiasNegadas, ct.ValorCompras7DiasNegadas, ct.ValorUltCompra, ct.DataUltCompra, ct.CodigoRespostaAutorizadorUltCompra, ct.PaisUltCompra, ct.Id_EstadoUlt, ct.UltFraudScore, ct.DataUltCompraAlertada,ct.DataDesbloqueio, QtdEventos,  (GETDATE() - ct.DataCriacao) AS TempoCartao, CASE WHEN tr.CodigoTipoResolucao = 1 then  'Fraude' ELSE 'Nao Fraude' end as Resolucao FROM fraude..compras co INNER JOIN FraudeAtend..filaEventos fl ON co.id_compra = fl.Id_Compra INNER JOIN FraudeAtend..TiposResolucao tr ON tr.Id_TipoResolucao = fl.Id_TipoResolucao INNER JOIN Fraude..Cartoes ct ON co.Id_Cartao = ct.Id_Cartao INNER JOIN (SELECT CO.ID_COMPRA as COMPRA, COUNT(*) AS QtdEventos FROM fraude..compras co INNER JOIN FraudeAtend..filaEventos fl ON co.id_compra = fl.Id_Compra INNER JOIN FraudeAtend..TiposResolucao tr ON tr.Id_TipoResolucao = fl.Id_TipoResolucao INNER JOIN Fraude..Cartoes ct ON co.Id_Cartao = ct.Id_Cartao WHERE co.id_emissor = "+ emissor +" GROUP BY co.id_compra ) AS QtdEventos on QtdEventos.COMPRA = co.Id_Compra WHERE ct.Id_Emissor = "+ emissor +" and co.id_emissor = "+ emissor +" and datacompra < dateadd(day,"+ date0 +",getdate()) and datacompra > dateadd(day,"+ date1 +",getdate()) and fl.Id_EventoCompra = (SELECT MAX(fev.Id_EventoCompra) FROM FraudeAtend..filaEventos fev WHERE fev.Id_Compra = co.Id_Compra)"

print("Carregando Dataframe...")
dataset_train = pd.read_sql(query,connection)

#Exclusão das colunas de correlação negativa, após visualização da matriz de correlação
dataset_train = dataset_train.drop(["QtdComprasDiaAprovadas", "QtdCompras3DiasAprovadas", 
									"QtdCompras7DiasAprovadas", "QtdComprasTotalAprovadas",
								    "ValorComprasDiaAprovadas", "ValorCompras3DiasAprovadas", 
								    "ValorCompras7DiasAprovadas"], axis = 1)

#corrigindo dados com NULL
print("Corrigindo colunas com dados nulos...")

dataset_train = data_correction(dataset_train)


'''Balanceamento, serve para manter a base equilibrada em termos de número de fraudes e não-fraudes,
   Para o Grid-Search, é altamente indicado que as classes estejam balanceadas.
'''
df0 = dataset_train[dataset_train['Resolucao'] == 'Fraude']
df1 = dataset_train[dataset_train['Resolucao'] == 'Nao Fraude']
df1 = df1.head(len(df0))
dataset_train = df1.append(df0)


#separando os dados da label de decisão
num_cols = len(dataset_train.columns)
X_train = dataset_train.iloc[:,:-1].values #altera as colunas para novos testes
Y_train = dataset_train.iloc[:, num_cols-1].values

################################ LABELS ################################

'''
	Gera hashs numéricas exclusivas para cada novo dado do dataframe,
	o algoritmo de classificação só observa dados numéricos.
'''
print("Codificando dados...")
labelencoder_x = LabelEncoder()
labelencoder_y = LabelEncoder()

for x in range(len(X_train[0])):
	X_train[:, x] = labelencoder_x.fit_transform(X_train[:, x])

Y_train = labelencoder_y.fit_transform(Y_train)

print("Salvando modelo da codificacao...")
#Salva modelo treinado no disco
joblib.dump(labelencoder_x, 'encoderx.pkl')
joblib.dump(labelencoder_y, 'encodery.pkl')


############################## NORMALIZAÇÃO #####################################

#Coloca todas as hashs geradas pelo labelenconder entre 0 e 1.

print("Normalizando dados...")
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

print("Salvando modelo da normalizacao...")
#Salva modelo treinado no disco
joblib.dump(sc_X, 'scaler.pkl')

print("Iniciando processamento do grid-search...")
############################## RANDON FOREST################################


classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)

t_range = list(range(1, 60))
# grid de parâmtros do random forest a serem testados.
param_grid = {"n_estimators": t_range,  # valores de parâmetros a serem testados
              "max_depth": [3, None],
              "max_features": [1,6],
              "min_samples_split": [2,6,8,10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}


print ("# RANDON FOREST #")
Classification_model_gridsearchCV("RandomForest", classifier,param_grid,X_train,Y_train)  

###################### SVM ###########################

classifier = SVC()
classifier.fit(X_train, Y_train)

# grid de parâmetros do svm 
param_grid = [
              {'C': [1, 10, 100, 1000], # valores de parâmetros a serem testados
               'kernel': ['linear']
              },
              {'C':  [1, 10, 100, 1000],
               'kernel': ['poly'],
               'degree': [2, 3],
               },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              },
			] 

print ("# SVM #")
Classification_model_gridsearchCV("SVM", classifier,param_grid,X_train,Y_train) 

###################### KNN ###########################

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, Y_train)

# grid de parâmtros do knn e variáveis do grid
k_range = list(range(1, 30))
leaf_size = list(range(1,30))
weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}

print ("# KNN #")
Classification_model_gridsearchCV("KNN", classifier,param_grid,X_train,Y_train)