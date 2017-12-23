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
query = "SELECT co.DataCompra, co.Valor, co.ValorDestino, co.ValorOrigem, co.Pais, co.Origem, co.CodigoRespostaFraude, co.MCC, co.CodigoMoedaOrigem, co.CodigoMoedaDestino, co.ModoEntradaTerminal, co.Id_Estado, co.ID_Produto, co.DataSaidaCompra, co.FraudScore, tr.Descricao, ct.UF, ct.LimiteGlobal, ct.LimiteGlobalCliente, ct.LimiteSaqueNacGlobal, ct.QtdComprasDiaAprovadas, ct.QtdCompras3DiasAprovadas, ct.QtdCompras7DiasAprovadas, ct.QtdComprasTotalAprovadas, ct.QtdComprasDiaNegadas, ct.QtdCompras3DiasNegadas, ct.QtdCompras7DiasNegadas, ct.QtdComprasTotalNegadas, ct.ValorComprasDiaAprovadas, ct.ValorCompras3DiasAprovadas, ct.ValorCompras7DiasAprovadas, ct.ValorComprasDiaNegadas, ct.ValorCompras3DiasNegadas, ct.ValorCompras7DiasNegadas, ct.ValorUltCompra, ct.DataUltCompra, ct.CodigoRespostaAutorizadorUltCompra, ct.PaisUltCompra, ct.Id_EstadoUlt, ct.UltFraudScore, ct.DataUltCompraAlertada,ct.DataDesbloqueio, QtdEventos,  (GETDATE() - ct.DataCriacao) AS TempoCartao, CASE WHEN tr.CodigoTipoResolucao = 1 then  'Fraude' ELSE 'Nao Fraude' end as Resolucao FROM fraude..compras co INNER JOIN FraudeAtend..filaEventos fl ON co.id_compra = fl.Id_Compra INNER JOIN FraudeAtend..TiposResolucao tr ON tr.Id_TipoResolucao = fl.Id_TipoResolucao INNER JOIN Fraude..Cartoes ct ON co.Id_Cartao = ct.Id_Cartao INNER JOIN (SELECT CO.ID_COMPRA as COMPRA, COUNT(*) AS QtdEventos FROM fraude..compras co INNER JOIN FraudeAtend..filaEventos fl ON co.id_compra = fl.Id_Compra INNER JOIN FraudeAtend..TiposResolucao tr ON tr.Id_TipoResolucao = fl.Id_TipoResolucao INNER JOIN Fraude..Cartoes ct ON co.Id_Cartao = ct.Id_Cartao WHERE co.id_emissor = "+ emissor +" GROUP BY co.id_compra ) AS QtdEventos on QtdEventos.COMPRA = co.Id_Compra WHERE ct.Id_Emissor = "+ emissor +" and co.id_emissor = "+ emissor +" and datacompra < dateadd(day,"+ date +",getdate()) and fl.Id_EventoCompra = (SELECT MAX(fev.Id_EventoCompra) FROM FraudeAtend..filaEventos fev WHERE fev.Id_Compra = co.Id_Compra)"

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



