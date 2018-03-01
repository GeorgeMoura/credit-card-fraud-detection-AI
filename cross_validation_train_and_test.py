# -*- coding: utf-8 -*-
"""
@authors: Gabriel Gonçalves, George Moura

Este código contém todo o treinamento, desde a query no banco até a classificação, cross-validation e exibição de resultados.
Os algoritmos de pré-processamento, como Grid-Search, Clusterização Hierarquica e Matriz de Correlação, devem ser executados antes,
em detrimento à escolha de parâmetros e de atributos deste código. Muito embora, os parâmetros e atributos deste código já passaram
por todo o pipeline de pré-processamento necessário, ou seja, configuração ótima para a análise feita.
"""

print("Importando bibliotecas...")

import numpy as np
import pandas as pd
import time
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

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

#Função responsável pelo treino do classificador, Cross-Validation, e matriz de confusão. Última função a ser chamada.
def classification(name, classifier, x_train, x_test, y_train, y_test, emissor):

	print("Treinando classificador com todos os dados de treino [sem cross validation]...")
	classifier.fit(x_train, y_train)

	menor = 0.0
	maior = 1.0
	#Cross-Validation
	print("Executando Cross-Validation...")
	scores = cross_val_score(classifier, X_train, Y_train, cv=5)
	for i in range(0,5):
	    print("Cross-Validation Score : %s" % "{0:.3%}".format(scores[i]))
	    if (menor < scores[i]):
	    	menor = scores[i]
	    if (maior > scores[i]):
	    	maior = scores[i]
	    
	print("\nIntervalo de confiança : " + str((menor-maior)*100) + "%\n")
	print("Cross-Validation Mean Score : %s" % "{0:.3%}\n".format(np.mean(scores)))

	#teste de predição
	y_pred = classifier.predict(x_test)
	        
	print("Gerando matriz de confusão...")
	#matriz de confusão
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, y_pred)

	print("\nInformacoes da classificacao:")
	accuracy = (cm[0,0] + cm[1,1])/len(y_test)
	FP = cm[1,0]
	FN = cm[0,1]
	print("Accuracy:")
	print(accuracy)
	stream = "Falso Positivo: " + str(FP) + " - Falso Negativo: " + str(FN)
	print(stream)
	print("Matriz de confusao:")
	print(cm)

	print("\nSalvando modelo do classificador...")
	#Salvando modelo do classificador
	model_name = name + '_' + emissor + '.pkl'
	print("Nome do modelo: " + model_name)
	joblib.dump(classifier, model_name)


################################ IMPORTANDO DADOS ######################

print("Conectando a base...")
connection = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=255.255.255.255;DATABASE=master;Trusted_Connection=Yes')

#Montagem da query, selecionar emissor e quantos dias à serem ignorados a partir de hoje.
emissor = "73"
date = "-1"
query = "SELECT..."

print("Carregando Dataframe...")
dataset_train = pd.read_sql(query,connection)

#Exclusão das colunas de correlação negativa, após visualização da matriz de correlação
dataset_train = dataset_train.drop(["..."], axis = 1)

#corrigindo dados com NULL
print("Corrigindo colunas com dados nulos...")

dataset_train = data_correction(dataset_train)


'''Balanceamento, serve para manter a base equilibrada em termos de número de fraudes e não-fraudes,
   muitas vezes necessário caso o número de não-fraudes seja extremamente superior ao número de fraudes,
   ou caso o seu dataframe seja muito grande. Em termos gerais, fica a critério de análise,
	Nos testes que fizemos, balanceamos o emissor x, e para o emissor y os testes com e sem
	balanceamento resultaram de maneira bastante semelhante.
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

print("Separando dataframe em treino e teste...")
#separando dados de treino e de teste, 33% do dataframe é carregado como parte de teste, necessário também, para gerar a matriz de confusão.
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.33, random_state = 0)


'''
	Carregamento dos algoritmos de classificação, os parâmetros estabelecidos
	foram gerados a partir de um Grid-Search previamente executado.
	Comentar as linhas referentes aos classificadores a serem ignorados.
'''
algorithm = "RandomForest"
#algorithm = "SVM"
#algorithm = "KNN"

if(algorithm == "RandomForest"):
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(criterion='entropy', max_depth=None, max_features=6, min_samples_leaf=1, min_samples_split=10, n_estimators=22)
elif(algorithm == "SVM"):
	from sklearn.svm import SVC
	classifier = SVC(C=1000, gamma=0.001, kernel='rbf')
elif(algorithm == "KNN"):
	from sklearn.neighbors import KNeighborsClassifier
	classifier = KNeighborsClassifier(leaf_size=1, n_neighbors=16, weights='distance')

classification(algorithm, classifier, x_train, x_test, y_train, y_test, emissor)
