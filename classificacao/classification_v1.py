#BRUNO EDUARDO FARIAS -  GRR20186715
#CDD 2021-2

import csv, sys, math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,make_scorer,classification_report,precision_recall_curve,RocCurveDisplay,auc
from sklearn.utils import resample, shuffle
from scipy import interp

def load_data(nome_arquivo, drop_label):
  input_data = pd.read_csv(nome_arquivo)

  row, col = input_data.shape
  logging.info(f'Linhas: {row} Colunas: {col}') 
  #ṕrint(input_data.head(10))

  #salva a classificação
  y = input_data['Class'].tolist()

  #remove a classificação (col Class)
  if drop_label == 1:
    input_data = input_data.drop('Class', axis = 1)
    logging.info('Após drop label: ')
    print(input_data.head(10)) 

  #retira o atributo Time por causa do resample
  input_data = input_data.drop('Time', axis = 1)

  input_data_scaled = input_data.copy()

  #normalizando dados
  #logging.info('Normalizando...')
  #input_data_scaled[input_data_scaled.columns] = StandardScaler().fit_transform(input_data_scaled)

  #logging.info(input_data_scaled.describe())

  return input_data_scaled, y

#plota a matriz de confusão em arquivo
def plot_matriz_conf(cm, alg):
  ax = sns.heatmap(cm, annot=True, cmap='Blues')

  ax.set_title('Matriz de confusão')
  ax.set_xlabel('Valores previstos')
  ax.set_ylabel('Valores atuais')

  ax.xaxis.set_ticklabels(['False', 'True'])
  ax.yaxis.set_ticklabels(['False', 'True'])

  #plt.show()
  i = 1
  nomeFig1 = 'matriz_conf_'+str(i)+'_'+alg+'.png'
  while os.path.exists(nomeFig1):
    i += 1
    nomeFig1 = 'matriz_conf_'+str(i)+'_'+alg+'.png'
  plt.savefig(nomeFig1)
  logging.info('Salvo em ' + nomeFig1)

  #limpa a figura para não plotar uma em cima da outra
  plt.clf()

  ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')

  ax.set_title('Matriz de confusão')
  ax.set_xlabel('Valores previstos')
  ax.set_ylabel('Valores atuais')

  ax.xaxis.set_ticklabels(['False', 'True'])
  ax.yaxis.set_ticklabels(['False', 'True'])

  #plt.show()

  i = 1
  nomeFig2 = 'matriz_conf_porc_'+str(i)+'_'+alg+'.png'
  while os.path.exists(nomeFig2):
    i += 1
    nomeFig2 = 'matriz_conf_porc_'+str(i)+'_'+alg+'.png'
  plt.savefig(nomeFig2)
  logging.info('Salvo em ' + nomeFig2)

  #limpa a figura para não plotar uma em cima da outra
  plt.clf()



def resample_f(data):
  logging.info('Fazendo resample dos dados...')

  #separa a classe de minoria (fraude) em um dataframe diferente
  df_1 = data[data['Class'] == 1]

  #logging.info('df_1>')
  #print(df_1.head(10))

  #separa a classe de maioria (não fraude) em um dataframe diferente
  other_df = data[data['Class'] != 1]

  #logging.info('other_df>')
  #print(other_df.head(10))

  #upsample a classe de minoria -> 1 amostra fraude para cada 5 amostras não fraude
  df_1_upsampled = resample(df_1, n_samples=int(len(other_df)/5))

  #concatena o dataframe upsampled
  df_1_upsampled = pd.concat([df_1_upsampled, other_df])

  #reseta index
  df_1_upsampled = df_1_upsampled.reset_index()

  #tira o index
  df_1_upsampled.drop(['index'], axis=1, inplace=True)
  
  logging.info('df_1_upsampled>')
  print(df_1_upsampled.head(10))

  #logging.info('df_1_upsampled>')
  #print(df_1_upsampled)

  d1_fraude_end = df_1_upsampled[df_1_upsampled['Class'] == 1]
  d1_nao_fraude_end = df_1_upsampled[df_1_upsampled['Class'] != 1]

  logging.info('Total de amostras SEM fraude: {}'.format(len(d1_nao_fraude_end)))
  logging.info('Total de amostras COM fraude: {}'.format(len(d1_fraude_end)))

  #salva as classificaçoes
  targets = df_1_upsampled['Class']

  #tira as classificacoes
  df_1_upsampled.drop(['Class'], axis=1, inplace=True)

  #logging.info('Ao final de resample: df_1_upsampled>')
  #print(df_1_upsampled.head(10))

  logging.info('Final do resample!')

  return df_1_upsampled, targets

#roda o GridSearchCV para encontrar o melhor classificador de acordo com o tipo de métrica escolhido
def grid_search_wrapper(clf, param_grid, scorers, X_train, X_test, y_train, y_test, refit_score='precision_score'):
  skf = StratifiedKFold(n_splits=10) #K-fold

  #estimator = clf
  #param_grid = parâmetros de classificaçao definidos pelo usuário
  #scoring = estratégias definidas pelo usuário para avaliar a performance dos modelos
  #refit = reajusta um classificador utilizando os melhores parametros do dataset
  #cv = estratégia de separação da validação em cruz  
  grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1)

  #treina os modelos
  grid_search.fit(X_train, y_train)

  #faz as predições
  y_pred = grid_search.predict(X_test)

  logging.info('Melhores parâmetros para {}'.format(refit_score))
  logging.info(grid_search.best_params_)

  logging.info('Matriz de confusão para os dados de teste: ')
  logging.info(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['pred_neg', 'pred_pos'], index=['neg','pos']))

  return grid_search

#funções auxiliares para geração de curvas ROC e P/R

#ajusta as predições das classes baseado no limite de previsão
def adjusted_classes(y_scores, t):
  return [1 if y >= t else 0 for y in y_scores]

#plota a curva de predição recall e mostra o valor atual para cada
#ao identificar o limite do classificador
def precision_recall_threshold(p, r, thresholds, y_scores, y_test, t=0.5):
  y_pred_adj = adjusted_classes(y_scores, t)
  print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj), 
                     columns=['pred_neg', 'pred_pos'],
                     index=['neg', 'pos']))

  #plota a curva
  plt.figure(figsize=(8,8))
  plt.title('Curva de precisão e recall')
  plt.step(r, p, step='post', alpha=0.2, color='b')
  plt.xlim([0.5, 1.01])
  plt.ylim([0.5, 1.01])
  plt.xlabel('Recall')
  plt.ylabel('Precisão')

  #plota o limite atual na linha
  close_default_clf = np.argmin(np.abs(thresholds - t))
  plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)

  nomeFig = 'curvas_'+alg+'.png'
  plt.savefig(nomeFig)
  logging.info('Salvo em ' + nomeFig2)

  #limpa a figura para não plotar uma em cima da outra
  plt.clf()


#gera as curvas ROC e P/R
def curvas(grid_search_clf, X_test, y_test, tipoAlg):
  logging.info('Gerando curvar para o algorimo ' + tipoAlg)
  y_scores = grid_search_clf.predict_proba(X_test)[:, 1]

  p, r, thresholds = precision_recall_curve(y_test, y_scores)

  precision_recall_threshold(p, r, thresholds, y_scores, y_test, 0.9)

def salvar_modelo(nome_algoritmo, descricao_modelo, modelo):
  logging.info('Salvando modelo ' + nome_algoritmo + ' - ' + descricao_modelo)
  nome_modelo = nome_algoritmo + "_" + descricao_modelo + ".sav"
  folder_modelos = 'modelos_treinados/'
  path_completo = folder_modelos + nome_modelo;

  if os.path.exists(path_completo):
    print('Arquivo ' + path_completo + ' já existe!')
    os.remove(path_completo)
    #exit()

  pickle.dump(modelo, open(path_completo, 'wb'))
  logging.info('Modelo salvo em ' + path_completo)

def randomForest2(data, targets, X_train, X_test, y_train, y_test, descricao_modelo, params):
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('Fazendo RandomForest (novo)')

  inicio = datetime.datetime.now();

  #pega os parâmetros (definidos pelo ajuste de limitar ou o default)
  best_n_estimators = params['n_estimators']
  best_min_samples_split = params['min_samples_split']
  best_max_depth = params['max_depth']
  best_max_features = params['max_features']

  logging.info('Rodando randomForest para as configuracoes escolhidas:')
  logging.info('n_estimators: {}'.format(best_n_estimators))
  logging.info('min_samples_split: {}'.format(best_min_samples_split))
  logging.info('max_depth: {}'.format(best_max_depth))
  logging.info('max_features: {}'.format(best_max_features))

  #cria classificador
  clf=RandomForestClassifier(n_estimators=best_n_estimators, 
                             min_samples_split=best_min_samples_split, 
                             max_depth=best_max_depth, 
                             max_features=best_max_features, 
                             n_jobs=-1)

  #treinar o modelo usando os sets y_pred=clf.predict(X_test)
  clf.fit(X_train,y_train)

  #aplicar nas amostras de teste
  y_pred=clf.predict(X_test)

  final = datetime.datetime.now();

  tempoPassado = final - inicio

  logging.info('Tempo passado: %d microsegundos | %d segundos' % (tempoPassado.microseconds, tempoPassado.seconds))

  #gravar os modelos para uso posterior
  salvar_modelo('RF', descricao_modelo, clf)

  #matriz de confusão
  cm = confusion_matrix(y_test, y_pred)

  logging.info("Confusion Matrix:")
  print(cm)

  #métrica de acurácia
  ac = accuracy_score(y_test,y_pred)

  logging.info("Accuracy score:")
  print(ac)

  #recall
  rc = recall_score(y_test, y_pred)

  logging.info("Recall:")
  print(rc)

  #matriz de confusão
  plot_matriz_conf(cm, 'RF')

  print(classification_report(y_test, y_pred))

  # y_pred_proba = clf.predict_proba(X_test)[::,1]
  # fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

  # plt.plot(fpr, tpr)
  # plt.ylabel('True Positive Rate')
  # plt.xlabel('False Positive Rate')
  # plt.show()

  #CURVA ROC COM KFOLD = 5
  curva_roc_kfold(clf, data, targets, X_train, y_train, X_test, y_test, 'RF', descricao_modelo)

def curva_roc_kfold(clf, data, targets, X_train, y_train, X_test, y_test, nome_algoritmo, descricao_modelo):
  logging.info('Gerando curva ROC com KFold de 5 para '+ nome_algoritmo + " - " + descricao_modelo)
  cv = StratifiedKFold(n_splits=5, shuffle=False)

  #limpa a figura para não plotar uma em cima da outra
  plt.clf()

  tprs = []
  aucs = []
  mean_fpr = np.linspace(0,1,100)
  i = 1
  for train,test in cv.split(data, targets):
    prediction = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, t = metrics.roc_curve(y_test, prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

  plt.plot([0,1], [0,1], linestyle='--',lw=2,color='black')
  mean_tpr = np.mean(tprs, axis=0)
  mean_auc = auc(mean_fpr, mean_tpr)
  plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.title('ROC')
  plt.legend(loc='lower right')
  plt.text(0.32,0.7,'More accurate area', fontsize=12)
  plt.text(0.63,0.4,'Less accurate area', fontsize=12)
  #plt.show()

  nomeFig = "imagens/curvas_ROC/ROC_" + nome_algoritmo + "_" + descricao_modelo + ".png"
  plt.savefig(nomeFig)
  logging.info('Salvo em ' + nomeFig)

  #limpa a figura para não plotar uma em cima da outra
  plt.clf()

def knn(data, targets, X_train, X_test, y_train, y_test, descricao_modelo, params, ajusteLimiar):

  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('Fazendo KNN (novo)')

  inicio = datetime.datetime.now();

  #pega os parâmetros (definidos pelo ajuste de limiar ou o default)
  best_n_neighbors = params['n_neighbors']

  #aplica o KNN com a melhor quantidade de vizinhos q o ajuste definiu
  logging.info('Rodando KNN para as configuracoes escolhidas...')
  logging.info('n_neighbors: {}'.format(best_n_neighbors))

  #cria classificador
  classifier = KNeighborsClassifier(n_neighbors = best_n_neighbors)

  #treina
  classifier.fit(X_train, y_train)

  #aplica sobre as amostras de teste
  y_pred = classifier.predict(X_test)

  final = datetime.datetime.now();

  tempoPassado = final - inicio

  logging.info('Tempo passado: %d microsegundos | %d segundos' % (tempoPassado.microseconds, tempoPassado.seconds))

  #gravar os modelos para uso posterior
  salvar_modelo('KNN', descricao_modelo, classifier)

  #matriz de confusão
  cm = confusion_matrix(y_test, y_pred)

  logging.info("Confusion Matrix:")
  logging.info(cm)

  #métrica de acurácia
  ac = accuracy_score(y_test,y_pred)

  logging.info("Accuracy score:")
  print (ac)

  #recall
  rc = recall_score(y_test, y_pred)

  logging.info("Recall:")
  logging.info(rc)

  plot_matriz_conf(cm, 'KNN')

  print(classification_report(y_test, y_pred))
  

def svm_f(data, targets, X_train, X_test, y_train, y_test, ajusteLimiar):
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('')
  logging.info('Fazendo SVM...')

  inicio = datetime.datetime.now();

  if ajusteLimiar:
    #fazendo ajuste de limiar de decisão
    logging.info('Rodando ajuste de limiar...')

    #dividir o dataset entre treinamento e validação
    logging.info('Separando dataset em teste e treinamento...')
    X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets)

    logging.info('======= Distribuiçao de classes (y_train) ===========')
    logging.info(y_train.value_counts(normalize=True))
    logging.info('')
    logging.info(y_train.value_counts())
    logging.info('======= Distribuiçao de classes (y_test) ===========')
    logging.info(y_test.value_counts(normalize=True))
    logging.info('')
    logging.info(y_test.value_counts())

    #normalizando as amostras
    logging.info('Normalizando as amostras...')
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #(100 árvores e todos os processadores/núcleos)
    clf=KNeighborsClassifier(n_jobs=-1)

    param_grid = {
      'n_neighbors': [1, 3, 5], #número de vizinhos
    }

    #apenas para acelerar os testes
    # param_grid = {
    #   'n_neighbors': [1], #número de vizinhos
    # }

    scorers = {
      'precision_score': make_scorer(precision_score), #habilidade de não classificar como positivo uma amostra que é negativa
      'recall_score': make_scorer(recall_score), #habilidade de encontrar todas as amostras positivas
      'accuracy_score': make_scorer(accuracy_score) # fração das previsões que o modelo acertou
    }

    inicioGridSearch = datetime.datetime.now();

    grid_search_clf = grid_search_wrapper(clf, param_grid, scorers, X_train, X_test, y_train, y_test, refit_score='precision_score')

    best_n_neighbors = grid_search_clf.best_params_['n_neighbors']
    logging.info('Melhor n_neighbors: {}'.format(best_n_neighbors))

    finalGridSearch = datetime.datetime.now();
    tempoPassadoGridSearch = finalGridSearch - inicioGridSearch
    logging.info('Tempo passado: %d microsegundos | %d segundos' % (tempoPassadoGridSearch.microseconds, tempoPassadoGridSearch.seconds))
    #curvas(grid_search_clf, X_test, y_test, 'SVM')

  logging.info('Fazendo SVM...')
  classifier = svm.SVC(kernel='linear')
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)

  final = datetime.datetime.now();

  tempoPassado = final - inicio

  logging.info('Tempo passado: %d microsegundos | %d segundos' % (tempoPassado.microseconds, tempoPassado.seconds))

  #matriz de confusão
  cm = confusion_matrix(y_test, y_pred)

  logging.info("Confusion Matrix:")
  logging.info(cm)

  #métrica de acurácia
  ac = accuracy_score(y_test,y_pred)

  logging.info("Accuracy score:")
  print (ac)

  #recall
  rc = recall_score(y_test, y_pred)

  logging.info("Recall:")
  logging.info(rc)

  plot_matriz_conf(cm, 'svm')

  print(classification_report(y_test, y_pred))
  
def ajusteLimiarRF(data, targets, descricao_modelo):
  #fazendo ajuste de limiar de decisão
  logging.info('Rodando ajuste de limiar...')

  #dividir o dataset entre treinamento e validação
  logging.info('Separando dataset em teste e treinamento...')
  X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets)

  logging.info('======= Distribuiçao de classes (y_train) ===========')
  print(y_train.value_counts(normalize=True))
  logging.info('')
  print(y_train.value_counts())
  logging.info('======= Distribuiçao de classes (y_test) ===========')
  print(y_test.value_counts(normalize=True))
  logging.info('')
  print(y_test.value_counts())

  #normalizando as amostras
  logging.info('Normalizando as amostras...')
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #todos os processadores/núcleos
  clf=RandomForestClassifier(n_jobs=-1)

  # param_grid = {
  #   'min_samples_split': [3, 5], #número mínimo de amostras necessárias para dividir um nó interno
  #   'n_estimators': [50, 100], #número de árvores
  #   'max_depth': [3, 15], #altura máxima da árvore
  #   'max_features': [5, 8] #número de atributos a considerar quando procurar a melhor divisão
  # }

  #apenas para acelerar os testes
  param_grid = {
    'min_samples_split': [3, 5], #número mínimo de amostras necessárias para dividir um nó interno
    'n_estimators': [5, 10], #número de árvores
    'max_depth': [1, 3], #altura máxima da árvore
    'max_features': [1, 3] #número de atributos a considerar quando procurar a melhor divisão
  }

  scorers = {
    'precision_score': make_scorer(precision_score), #habilidade de não classificar como positivo uma amostra que é negativa
    'recall_score': make_scorer(recall_score), #habilidade de encontrar todas as amostras positivas
    'accuracy_score': make_scorer(accuracy_score) # fração das previsões que o modelo acertou
  }

  inicioGridSearch = datetime.datetime.now();

  logging.info('Rodando GridSearchCV...')
  grid_search_clf = grid_search_wrapper(clf, param_grid, scorers, X_train, X_test, y_train, y_test, refit_score='precision_score')
  
  best_n_estimators = grid_search_clf.best_params_['n_estimators']
  logging.info('Melhor n_estimators: {}'.format(best_n_estimators))

  best_min_samples_split = grid_search_clf.best_params_['min_samples_split']
  logging.info('Melhor min_samples_split: {}'.format(best_min_samples_split))

  best_max_depth = grid_search_clf.best_params_['max_depth']
  logging.info('Melhor max_depth: {}'.format(best_max_depth))

  best_max_features = grid_search_clf.best_params_['max_features']
  logging.info('Melhor max_features: {}'.format(best_max_features))

  finalGridSearch = datetime.datetime.now();
  tempoPassadoGridSearch = finalGridSearch - inicioGridSearch
  logging.info('Tempo passado: %d microsegundos | %d segundos' % (tempoPassadoGridSearch.microseconds, tempoPassadoGridSearch.seconds))
  #curvas(grid_search_clf, X_test, y_test, 'RF')

  salvar_modelo('RF', descricao_modelo, modelo=RandomForestClassifier(n_estimators=best_n_estimators, 
                             min_samples_split=best_min_samples_split, 
                             max_depth=best_max_depth, 
                             max_features=best_max_features, 
                             n_jobs=-1))

  return grid_search_clf.best_params_

def ajusteLimiarKNN(data, targets, descricao_modelo):
  #fazendo ajuste de limiar de decisão
  logging.info('Rodando ajuste de limiar...')

  #dividir o dataset entre treinamento e validação
  logging.info('Separando dataset em teste e treinamento...')
  X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets)

  logging.info('======= Distribuiçao de classes (y_train) ===========')
  logging.info(y_train.value_counts(normalize=True))
  logging.info('')
  logging.info(y_train.value_counts())
  logging.info('======= Distribuiçao de classes (y_test) ===========')
  logging.info(y_test.value_counts(normalize=True))
  logging.info('')
  logging.info(y_test.value_counts())

  #normalizando as amostras
  logging.info('Normalizando as amostras...')
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #todos os processadores/núcleos
  clf=KNeighborsClassifier(n_jobs=-1)

  # param_grid = {
  #   'n_neighbors': [1, 3, 5], #número de vizinhos
  # }

  #apenas para acelerar os testes
  param_grid = {
   'n_neighbors': [1], #número de vizinhos
  }

  scorers = {
    'precision_score': make_scorer(precision_score), #habilidade de não classificar como positivo uma amostra que é negativa
    'recall_score': make_scorer(recall_score), #habilidade de encontrar todas as amostras positivas
    'accuracy_score': make_scorer(accuracy_score) # fração das previsões que o modelo acertou
  }

  inicioGridSearch = datetime.datetime.now();

  grid_search_clf = grid_search_wrapper(clf, param_grid, scorers, X_train, X_test, y_train, y_test, refit_score='precision_score')

  best_n_neighbors = grid_search_clf.best_params_['n_neighbors']
  logging.info('Melhor n_neighbors: {}'.format(best_n_neighbors))

  finalGridSearch = datetime.datetime.now();
  tempoPassadoGridSearch = finalGridSearch - inicioGridSearch
  logging.info('Tempo passado: %d microsegundos | %d segundos' % (tempoPassadoGridSearch.microseconds, tempoPassadoGridSearch.seconds))
  #curvas(grid_search_clf, X_test, y_test, 'KNN')

  salvar_modelo('KNN', descricao_modelo, modelo=KNeighborsClassifier(n_neighbors=best_n_neighbors, n_jobs=-1))

  return grid_search_clf.best_params_

def exporacao():
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(funcName)s()] %(message)s",
    handlers=[
      logging.FileHandler("debug.log"),
      logging.StreamHandler()
    ]
  )
  logging.info('Iniciando exploração do dataset...')

  #abrindo dataset que veio pelo sys.argv[1]
  nome_arquivo = sys.argv[1]

  logging.info('1. Lendo dados %s...' % (nome_arquivo))
  data_scaled, y = load_data(nome_arquivo, 0)

  #resample
  logging.info('2. Fazendo resample (3 conjuntos)...')
  data_scaled1, targets1 = resample_f(data_scaled)
  data_scaled2, targets2 = resample_f(data_scaled)
  data_scaled3, targets3 = resample_f(data_scaled)

  #GridSearch pra cada algoritmo e cada conjunto
  logging.info('3. Rodando ajuste de limiar para cada conjunto e algotimo...')

  #RF
  logging.info('3.1 Rodando ajuste de RandomForest para conjunto 1')
  params_rf_c1=ajusteLimiarRF(data_scaled1, targets1, 'c1')
  logging.info('3.2 Rodando ajuste de RandomForest para conjunto 2')
  params_rf_c2=ajusteLimiarRF(data_scaled2, targets2, 'c2')
  logging.info('3.3 Rodando ajuste de RandomForest para conjunto 3')
  params_rf_c3=ajusteLimiarRF(data_scaled3, targets3, 'c3')

  #KNN
  # logging.info('3.1 Rodando ajuste de KNN para conjunto 1')
  # params_knn_c1=ajusteLimiarKNN(data_scaled1, targets1, 'c1')
  # logging.info('3.2 Rodando ajuste de KNN para conjunto 2')
  # params_knn_c2=ajusteLimiarKNN(data_scaled2, targets2, 'c2')
  # logging.info('3.3 Rodando ajuste de KNN para conjunto 3')
  # params_knn_c3=ajusteLimiarKNN(data_scaled3, targets3, 'c3')


  #Com base nos melhores parâmetros gerados para cada algoritmo:

  #Treine modelos usando separação do DATASET em 50/50 e 80/20;
  logging.info('3. Separando os conjuntos resampled do DATASET em 50/50 e 80/20...')
  X_train11, X_test11, y_train11, y_test11 = train_test_split(data_scaled1, targets1, test_size = 0.5) #50/50
  X_train12, X_test12, y_train12, y_test12 = train_test_split(data_scaled2, targets2, test_size = 0.5) #50/50
  X_train13, X_test13, y_train13, y_test13 = train_test_split(data_scaled3, targets3, test_size = 0.5) #50/50

  X_train21, X_test21, y_train21, y_test21 = train_test_split(data_scaled1, targets1, test_size = 0.2) #80/20
  X_train22, X_test22, y_train22, y_test22 = train_test_split(data_scaled2, targets2, test_size = 0.2) #80/20
  X_train23, X_test23, y_train23, y_test23 = train_test_split(data_scaled3, targets3, test_size = 0.2) #80/20

  #normalizando as amostras
  logging.info('4. Normalizando as amostras...')
  sc = StandardScaler()

  #50/50
  X_train11 = sc.fit_transform(X_train11)
  X_test11 = sc.transform(X_test11)
  X_train12 = sc.fit_transform(X_train12)
  X_test12 = sc.transform(X_test12)
  X_train13 = sc.fit_transform(X_train13)
  X_test13 = sc.transform(X_test13)

  #80/20
  X_train21 = sc.fit_transform(X_train21)
  X_test21 = sc.transform(X_test21)
  X_train22 = sc.fit_transform(X_train22)
  X_test22 = sc.transform(X_test22)
  X_train23 = sc.fit_transform(X_train23)
  X_test23 = sc.transform(X_test23)

  #melhores parâmetros do randomForest gerados em execuções anteriores do ajuste de limiar
  best_n_estimators = 50
  best_min_samples_split = 5
  best_max_depth = 15
  best_max_features = 5
  params_RF_default = {"n_estimators": best_n_estimators, "min_samples_split": best_min_samples_split,
                    "max_features": best_max_features, "max_depth": best_max_depth}

  logging.info('5.  Fazendo RandomForest:')
  logging.info('5.1 Fazendo RandomForest com 50/50 - conjunto 1: ')
  randomForest2(data_scaled1, targets1, X_train11, X_test11, y_train11, y_test11, '50-50-c1', params_rf_c1)
  logging.info('5.2 Fazendo RandomForest com 50/50 - conjunto 2: ')
  randomForest2(data_scaled2, targets2, X_train12, X_test12, y_train12, y_test12, '50-50-c2', params_rf_c2)
  logging.info('5.3 Fazendo RandomForest com 50/50 - conjunto 3: ')
  randomForest2(data_scaled3, targets3, X_train13, X_test13, y_train13, y_test13, '50-50-c3', params_rf_c3)

  logging.info('5.4 Fazendo RandomForest com 80/20 - conjunto 1: ')
  randomForest2(data_scaled1, targets1, X_train21, X_test21, y_train21, y_test21, '80-20-c1', params_rf_c1)
  logging.info('5.5 Fazendo RandomForest com 80/20 - conjunto 2: ')
  randomForest2(data_scaled2, targets2, X_train22, X_test22, y_train22, y_test22, '80-20-c2', params_rf_c2)
  logging.info('5.6 Fazendo RandomForest com 80/20 - conjunto 3: ')
  randomForest2(data_scaled3, targets3, X_train23, X_test23, y_train23, y_test23, '80-20-c3', params_rf_c3)

  #melhores parâmetros do KNN gerados em execuções anteriores do ajuste de limiar
  best_n_neighbors = 3
  params_KNN_default = {"n_neighbors": best_n_neighbors}

  logging.info('6.  Fazendo KNN...')
  logging.info('6.1 Fazendo KNN com 50/50 - conjunto 1: ')
  knn(data_scaled1, targets1, X_train11, X_test11, y_train11, y_test11, '50-50-c1', params_knn_c1, True)
  logging.info('6.2 Fazendo KNN com 50/50 - conjunto 2: ')
  knn(data_scaled2, targets2, X_train12, X_test12, y_train12, y_test12, '50-50-c2', params_knn_c2, True)
  logging.info('6.3 Fazendo KNN com 50/50 - conjunto 3: ')
  knn(data_scaled3, targets3, X_train13, X_test13, y_train13, y_test13, '50-50-c3', params_knn_c3, True)

  logging.info('6.4 Fazendo KNN com 80/20 - conjunto 1: ')
  knn(data_scaled1, targets1, X_train21, X_test21, y_train21, y_test21, '80-20-c1', params_knn_c1, True)
  logging.info('6.5 Fazendo KNN com 80/20 - conjunto 2: ')
  knn(data_scaled2, targets2, X_train22, X_test22, y_train22, y_test22, '80-20-c2', params_knn_c2, True)
  logging.info('6.6 Fazendo KNN com 80/20 - conjunto 3: ')
  knn(data_scaled3, targets3, X_train23, X_test23, y_train23, y_test23, '80-20-c2', params_knn_c2, True)

  logging.info('7.  Fazendo SVM...')
  logging.info('7.1 Fazendo SVM com 50/50 - conjunto 1: ')
  svm_f(data_scaled1, targets1, X_train11, X_test11, y_train11, y_test11, False)
  logging.info('7.2 Fazendo SVM com 50/50 - conjunto 2: ')
  svm_f(data_scaled2, targets2, X_train12, X_test12, y_train12, y_test12, False)
  logging.info('7.3 Fazendo SVM com 50/50 - conjunto 3: ')
  svm_f(data_scaled3, targets3, X_train13, X_test13, y_train13, y_test13, False)

  logging.info('7.4 Fazendo SVM com 80/20 - conjunto 1: ')
  svm_f(data_scaled1, targets1, X_train21, X_test21, y_train21, y_test21, False)
  logging.info('7.4 Fazendo SVM com 80/20 - conjunto 2: ')
  svm_f(data_scaled2, targets2, X_train22, X_test22, y_train22, y_test22, False)
  logging.info('7.4 Fazendo SVM com 80/20 - conjunto 3: ')
  svm_f(data_scaled3, targets3, X_train23, X_test23, y_train23, y_test23, False)

if __name__ == "__main__":
  exporacao()