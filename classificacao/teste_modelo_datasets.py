import logging, sys, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,make_scorer,classification_report,precision_recall_curve,RocCurveDisplay,auc,f1_score
from sklearn.preprocessing import StandardScaler

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

  input_data_scaled = input_data.copy()

  return input_data_scaled, y

def testar_modelo():
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(funcName)s()] %(message)s",
    handlers=[
      logging.FileHandler("testes_modelos.log"),
      logging.StreamHandler()
    ]
  )
  logging.info('Iniciando testes de modelos...')

  #abrindo dataset que veio pelo sys.argv[1]
  nome_arquivo = sys.argv[1]

  logging.info('1. Lendo dados %s...' % (nome_arquivo))
  data_scaled, y = load_data(nome_arquivo, 0)

  #tira a classificação
  data_scaled = data_scaled.drop('Class', axis = 1)
  if('Time' in data_scaled):
    data_scaled = data_scaled.drop('Time', axis = 1)

  logging.info('2. Separando amostras de testes...')
  X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size = 0.5) #50/50

  #normalizando as amostras
  logging.info('Normalizando as amostras...')
  
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #carrega modelos
  logging.info('3. Carregando modelo...')
  rf_50_50_c3 = pickle.load(open('modelos_treinados/RF_50-50-c3.sav', 'rb'))
  knn_80_20_c3 = pickle.load(open('modelos_treinados/KNN_80-20-c3.sav', 'rb'))
  svm_80_20_c2 = pickle.load(open('modelos_treinados/SVM_80-20-c2.sav', 'rb'))

  modelos = {
    "rf": rf_50_50_c3,
    "knn": knn_80_20_c3,
    "svm": svm_80_20_c2
  }

  logging.info('4. Fazendo predições...')

  for key in modelos:
    logging.info(key + ":")
    y_pred = modelos[key].predict(X_test)

    logging.info(y_pred)

    #matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Confusion Matrix:")
    logging.info(cm)
    print(cm)

    #precisao
    pr = precision_score(y_test, y_pred)
    logging.info("Precision: {}".format(pr))

    #métrica de acurácia
    ac = accuracy_score(y_test,y_pred)
    logging.info("Accuracy: {}".format(ac))

    #recall
    rc = recall_score(y_test, y_pred)
    logging.info("Recall: {}".format(rc))

    #f1_score
    f1 = f1_score(y_test, y_pred)
    logging.info("f1_score: {}".format(f1))

  #y_pred = knn_80_20_c3.predict(X_test)
  #logging.info(y_pred)

  #matriz de confusão
  # cm = confusion_matrix(y_test, y_pred)
  # logging.info("Confusion Matrix:")
  # logging.info(cm)
  # print(cm)

  # #precisao
  # pr = precision_score(y_test, y_pred)
  # logging.info("Precision: {}".format(pr))

  # #métrica de acurácia
  # ac = accuracy_score(y_test,y_pred)
  # logging.info("Accuracy: {}".format(ac))

  # #recall
  # rc = recall_score(y_test, y_pred)
  # logging.info("Recall: {}".format(rc))

  # #f1_score
  # f1 = f1_score(y_test, y_pred)
  # logging.info("f1_score: {}".format(f1))


if __name__ == "__main__":
  testar_modelo()