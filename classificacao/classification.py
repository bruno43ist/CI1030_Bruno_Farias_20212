#BRUNO EDUARDO FARIAS -  GRR20186715
#CDD 2021-2

import csv, sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score

def load_data(nome_arquivo, drop_label):
	input_data = pd.read_csv(nome_arquivo)

	row, col = input_data.shape
	print(f'Linhas: {row} Colunas: {col}') 
	print(input_data.head(10))

	#salva a classificação
	y = input_data['Class'].tolist()

	#remove a classificação (col Class)
	if drop_label == 1:
		input_data = input_data.drop('Class', axis = 1)
		print('Após drops: ')
		print(input_data.head(10))	

	input_data_scaled = input_data.copy()

	#normalizando dados
	#print('Normalizando...')
	#input_data_scaled[input_data_scaled.columns] = StandardScaler().fit_transform(input_data_scaled)

	#print(input_data_scaled.describe())

	return input_data_scaled, y

def pca_embeddings(data_scaled):
	pca_2 = PCA(n_components=2)
	pca_2_result = pca_2.fit_transform(data_scaled)

	#imprime dados sobre o PCA
	#dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=data_scaled.columns, index=['PC_1', 'PC_2'])
	#print('\n\n', dataset_pca)
    
	#print('Componente principal 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
	#print('\n\nComponente principal 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())

	return pca_2_result, pca_2


def plotar_resultados(pca_result, label, centroids_pca, kmeans):
	h = 0.02

	x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
	y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(
		Z,
		interpolation="nearest",
		extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		cmap=plt.cm.Paired,
		aspect="auto",
		origin="lower",
	)

	plt.plot(pca_result[:, 0], pca_result[:, 1], "k.", markersize=2)
	centroids = kmeans.cluster_centers_
	plt.scatter(
		centroids_pca[:, 0],
		centroids_pca[:, 1],
		marker="x",
		s=169,
		linewidths=3,
		color="w",
		zorder=10,
	)

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()

def plot_matriz_conf(cm):
	ax = sns.heatmap(cm, annot=True, cmap='Blues')

	ax.set_title('Matriz de confusão')
	ax.set_xlabel('Valores previstos')
	ax.set_ylabel('Valores atuais')

	ax.xaxis.set_ticklabels(['False', 'True'])
	ax.yaxis.set_ticklabels(['False', 'True'])

	plt.show()

	ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')

	ax.set_title('Matriz de confusão')
	ax.set_xlabel('Valores previstos')
	ax.set_ylabel('Valores atuais')

	ax.xaxis.set_ticklabels(['False', 'True'])
	ax.yaxis.set_ticklabels(['False', 'True'])

	plt.show()

def randomForest(data):

	print('\n\n\n\n\n\nFazendo RandomForest')

	X = data.iloc[:, :30].values
	y = data.iloc[:,30].values
	print('X:')
	print(X)
	print('y:')
	print(y)

	inicio = datetime.datetime.now();

	#dividir o dataset entre treinamento e validação
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% treinamento and 30% testes

	#normalizando as amostras
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	#cria um classificador gaussiano
	#clf=RandomForestClassifier(n_estimators=100)
	clf=RandomForestClassifier(n_estimators=100)

	#treinar o modelo usando os sets y_pred=clf.predict(X_test)
	clf.fit(X_train,y_train)

	y_pred=clf.predict(X_test)

	final = datetime.datetime.now();

	tempoPassado = final - inicio

	print('Tempo passado: %d microsegundos | %d segundos' % (tempoPassado.microseconds, tempoPassado.seconds))

	#matriz de confusão
	cm = confusion_matrix(y_test, y_pred)

	print("Confusion Matrix:")
	print(cm)

	#métrica de acurácia
	ac = accuracy_score(y_test,y_pred)

	print("Accuracy score:",)
	print (ac)

	#recall
	rc = recall_score(y_test, y_pred)

	print("Recall:")
	print(rc)

	plot_matriz_conf(cm)

def knn(data):

	print('\n\n\n\n\n\nFazendo KNN')

	print(data.head(10))

	X = data.iloc[:, :30].values
	y = data.iloc[:,30].values
	print('X:')
	print(X)
	print('y:')
	print(y)

	inicio = datetime.datetime.now();

	#dividir o dataset entre treinamento e validação
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% treinamento and 30% testes

	#normalizando as amostras
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	#aplica o KNN com 5 vizinhos
	#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
	classifier = KNeighborsClassifier(n_neighbors = 5)
	classifier.fit(X_train, y_train)

	#testar com os 30% separados para teste
	y_pred = classifier.predict(X_test)

	final = datetime.datetime.now();

	tempoPassado = final - inicio

	print('Tempo passado: %d microsegundos | %d segundos' % (tempoPassado.microseconds, tempoPassado.seconds))

	#matriz de confusão
	cm = confusion_matrix(y_test, y_pred)

	print("Confusion Matrix:")
	print(cm)

	#métrica de acurácia
	ac = accuracy_score(y_test,y_pred)

	print("Accuracy score:",)
	print (ac)

	#recall
	rc = recall_score(y_test, y_pred)

	print("Recall:")
	print(rc)

	plot_matriz_conf(cm)
	


def exporacao():
	print('Iniciando exploração do dataset...')

	#abrindo dataset que veio pelo sys.argv[1]
	nome_arquivo = sys.argv[1]

	print('1. Lendo dados %s...' % (nome_arquivo))
	data_scaled, y = load_data(nome_arquivo, 0)

	#data_scaled = data_scaled.to_numpy()

	randomForest(data_scaled)

	knn(data_scaled)

if __name__ == "__main__":
	exporacao()