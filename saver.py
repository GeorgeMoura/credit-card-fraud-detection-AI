# -*- coding: utf-8 -*-
"""
@authors: Gabriel Gonçalves, George Moura

Código exemplo de como gerar modelos externos, e como carrega-los posteriormente.
Além disso, como carregar um dataframe a partir de um csv, sem precisar fazer uma consulta ao banco.
"""
from sklearn.externals import joblib
import pandas as pd

#Salva modelo treinado no disco
joblib.dump(classifier, 'model.pkl')

#Carrega modelo salvo para ser usado na classificação
classifier = joblib.load('model.pkl') 

'''exemplo de como carregar o dataframe a partir de um csv (no need 4 sql query!),
   agurmentos sep e decimal servem para definir o separador e o char que caracteriza números decimais, respectivamente.
'''
dataset_train = pd.read_csv("csv_example.csv", sep=';', decimal=',')