# Importação das bibliotecas necessárias
import pandas as pd  # Biblioteca para manipulação de dados
import numpy as np  # Biblioteca para operações matemáticas
from sklearn.feature_extraction.text import TfidfVectorizer  # Transforma textos em vetores numéricos
from sklearn.ensemble import RandomForestClassifier  # Algoritmo de classificação Random Forest
import pickle  # Serialização e desserialização de objetos Python
import nltk  # Ferramenta de processamento de linguagem natural
from nltk.tokenize import word_tokenize  # Função para dividir um texto em palavras

# Baixando o pacote 'punkt' do NLTK, necessário para a tokenização de texto
nltk.download('punkt')

# Carregando um arquivo Excel em um DataFrame pandas
df = pd.read_excel("br-sem-acentos.xlsx", engine='openpyxl')

# Concatenando todas as células da primeira coluna do DataFrame em uma única string, separando-as por espaço
texto = df.iloc[:, 0].str.cat(sep=' ')

# Selecionando a coluna 'Palavra' do DataFrame
texto_unico = df.loc[:, 'Palavra']

# Convertendo a série 'texto_unico' em um DataFrame
texto_unico_df = texto_unico.to_frame()

# Inicializando uma lista vazia para armazenar os textos tokenizados
tokenized_texts = []

# Iterando sobre cada linha do DataFrame 'texto_unico_df'
for index, row in texto_unico_df.iterrows():
    # Tokenizando cada célula na linha atual e adicionando a linha tokenizada à lista 'tokenized_texts'
    tokenized_row = [word_tokenize(cell, language='portuguese') for cell in row]
    tokenized_texts.append(tokenized_row)

# Convertendo a lista 'tokenized_texts' em um DataFrame
texto_unico_tokens = pd.DataFrame(tokenized_texts)

# Expandido cada lista dentro do DataFrame 'texto_unico_tokens' e empilhando os valores resultantes em uma única série
texto_unico_tokens_flat = texto_unico_tokens.apply(pd.Series.explode).stack()

# Criando um objeto TfidfVectorizer e transformando os textos tokenizados e aplanados em vetores TF-IDF
vectorizer = TfidfVectorizer()
texto_unico_tokens_flat_vectorizer = vectorizer.fit_transform(texto_unico_tokens_flat)

# Preparando os rótulos para a classificação múltipla, selecionando cada coluna do DataFrame que começa com uma letra do alfabeto
labels = []
for i in range(ord('a'), ord('z') + 1):
    letra = chr(i)
    column_name = f'{letra}'
    labels.append(df[column_name])

# Convertendo a lista de rótulos em um array numpy e ajustando sua forma para se adequar à estrutura de saída múltipla
y = np.array(labels)
y = y.reshape(-1, 26)

# Criando um objeto RandomForestClassifier e treinando o modelo com os vetores TF-IDF e os rótulos
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(texto_unico_tokens_flat_vectorizer, y)

# Serializando o modelo treinado e salvando em um arquivo
with open('modelo_random_forest.pkl', 'wb') as file:
    pickle.dump(model, file)
