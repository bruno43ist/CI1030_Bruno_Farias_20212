import logging, sys
import pandas as pd
from sklearn.utils import resample

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

  df_1_upsampled = resample(df_1, n_samples=int(len(other_df)/5), random_state=42)

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
  #df_1_upsampled.drop(['Class'], axis=1, inplace=True)

  #logging.info('Ao final de resample: df_1_upsampled>')
  #print(df_1_upsampled.head(10))

  logging.info('Final do resample!')

  return df_1_upsampled, targets

def gerarAmostras():
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(funcName)s()] %(message)s",
    handlers=[
      logging.FileHandler("gerador_amostras.log"),
      logging.StreamHandler()
    ]
  )
  logging.info('Iniciando geração de amostras...')

  #abrindo dataset que veio pelo sys.argv[1]
  nome_arquivo = sys.argv[1]

  logging.info('1. Lendo dados %s...' % (nome_arquivo))
  data_scaled, y = load_data(nome_arquivo, 0)

  #resample
  logging.info('2. Fazendo resample...')
  data_scaled1, targets1 = resample_f(data_scaled)

  data_scaled1.to_csv('datasets_gerados/dataset_42.csv', index=False)

if __name__ == "__main__":
  gerarAmostras()