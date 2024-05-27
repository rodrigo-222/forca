import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle

# Load and preprocess data
df = pd.read_excel("br-sem-acentos.xlsx", engine='openpyxl')
texto = df.iloc[:, 0].str.cat(sep=' ')
texto_unico = df['Palavra']

# Encode words using CountVectorizer
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))
X = vectorizer.fit_transform(texto.split()).toarray()

# Prepare labels for multi-output classification
labels = []
for i in range(ord('a'), ord('z') + 1):
    letra = chr(i)
    column_name = f'{letra}'
    labels.append(df[column_name])
y = np.array(labels)

# Reshape y to fit the multi-output structure
y = y.reshape(-1, 26)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a list to store models for each output
models = []

# Train a model for each output using LogisticRegression
for i in range(y_train.shape[1]):
    print(i)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train[:, i])
    models.append(model)

for i, model in enumerate(models):
    # Gerar um nome de arquivo único para cada modelo
    filename = f'model_{chr(ord("a")+i)}.pkl'
    print(filename)
    # Salvar o modelo em um arquivo separado
    with open(filename, 'wb') as file:
        model = pickle.dump(model, file)

# Wrap the models with MultiOutputClassifier
multi_output_classifier = MultiOutputClassifier(LogisticRegression(max_iter=500, random_state=42), n_jobs=len(models))

# Fit the classifier
multi_output_classifier.fit(texto_unico, y)

# Save the final model
filename = 'multi_output_logistic_regression.pkl'
with open(filename, 'wb') as file:
    pickle.dump(multi_output_classifier, file)
# Predict on the test set
y_pred = multi_output_classifier.predict(X_test)

# Calculate the accuracy score
print(X_test)
print(y_pred)

# Calcular a precisão para cada saída individualmente, usando zero_division=0
precision_scores = [precision_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0) 
                    for i in range(y_test.shape[1])]

# Calcular a recall para cada saída individualmente
recall_scores = [recall_score(y_test[:, i], y_pred[:, i], average='weighted') for i in range(y_test.shape[1])]

# Calcular a F1-score para cada saída individualmente
f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='weighted') for i in range(y_test.shape[1])]

# Calcular a média ponderada das precisões, recalls e F1-scores
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print(f"Média Ponderada da Precisão: {mean_precision}")
print(f"Média Ponderada da Recall: {mean_recall}")
print(f"Média Ponderada da F1-Score: {mean_f1}")
