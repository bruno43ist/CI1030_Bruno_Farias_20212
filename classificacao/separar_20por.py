import csv, sys, math
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from os.path import exists

if __name__ == "__main__":
	print('Separando 20% do dataset')

	nome_arquivo = sys.argv[1]
	input_data = pd.read_csv(nome_arquivo)

	header = input_data.columns

	#classes
	targets = input_data['Class']

	X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size = 0.2) #80/20

	#80%
	print('Len X_train: {}'.format(len(X_train)))
	print('Len y_train: {}'.format(len(y_train)))

	#20%
	print('Len X_test: {}'.format(len(X_test)))
	print('Len y_test: {}'.format(len(y_test)))

	oitenta_fraude = X_train[X_train['Class'] == 1]
	oitenta_nao_fraude = X_train[X_train['Class'] != 1]

	vinte_fraude = X_test[X_test['Class'] == 1]
	vinte_nao_fraude = X_test[X_test['Class'] != 1]

	print('80% tem {} fraudes e {} não fraudes'.format(len(oitenta_fraude), len(oitenta_nao_fraude)))
	print('20% tem {} fraudes e {} não fraudes'.format(len(vinte_fraude), len(vinte_nao_fraude)))

	#print('cont = {}'.format(cont))
	nome_arquivo_80 = 'creditCard-80.csv'

	if(os.path.exists(nome_arquivo_80)):
		print('Arquivo de 80% já existe!')
		exit()

	nome_arquivo_20 = 'creditCard-20.csv'

	if(os.path.exists(nome_arquivo_20)):
		print('Arquivo de 20% já existe!')
		exit()

	#exporta 80%
	X_train.to_csv(nome_arquivo_80, index=False)

	#exporta 20%
	X_test.to_csv(nome_arquivo_20, index=False)

	print('FIM!')

