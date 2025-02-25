import os
import cv2
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------------------------------------
# Caminhos das pastas e arquivos de labels
train_path = "BrainCancerDetection/dataset/X_train"
test_path = "BrainCancerDetection/dataset/X_test"
labels_train_file = "BrainCancerDetection/dataset/y_train/y_train.txt"
labels_test_file = "BrainCancerDetection/dataset/y_test/y_test.txt"

# Cria pastas para salvar os resultados, os modelos e os 'historys'
results_folder = "KMLPresults"
models_folder = "KMLPmodels"
historys_folder = "BrainCancerDetection/Historicos/KMLP"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if not os.path.exists(historys_folder):
    os.makedirs(historys_folder)

# Define um unico valor de seed
Seed = 404
tf.random.set_seed(seed = Seed)
np.random.seed(seed = Seed)
random.seed(Seed)

# Função para carregar as imagens e as labels
def load_dataset(image_dir, label_file, color_mode, img_size=(60, 60)):
    images = []
    labels = []

    # Carrega as labels do arquivo
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Carrega o nome da imagem e a label referente
        img_name, label = line.strip().split()
        img_path = os.path.join(image_dir, img_name)

        # Carrega a imagem de acordo com o filtro escolhido, reescala e normaliza
        if color_mode == "grayscale":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            img = img.astype("float32") / 255.0
            img = img.reshape(img_size[0], img_size[1], 1)

        elif color_mode == "rgb":
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype("float32") / 255.0 

        elif color_mode == "negative":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            img = 1.0 - (img.astype("float32") / 255.0)
            img = img.reshape(img_size[0], img_size[1], 1)
        
        elif color_mode == "hsv":
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.resize(img, img_size)
            img = img.astype("float32") / 255.0

        elif color_mode == "binary":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            img = cv2.resize(img, img_size)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = img.astype("float32") / 255.0
            img = img.reshape(img_size[0], img_size[1], 1)

        elif color_mode == "blur":
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, img_size)
            img = cv2.GaussianBlur(img, (15, 15), 0)
            img = img.astype("float32") / 255.0

        elif color_mode == "canny":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            img = cv2.Canny(img, 100, 200) 
            img = img.astype("float32") / 255.0
            img = img.reshape(img_size[0], img_size[1], 1)


        else:
            raise ValueError("Color mode not supported. Use 'grayscale', 'rgb', 'negative', 'hsv', 'binary', 'blur' or 'canny'.")

        images.append(img)
        labels.append(int(label))

    # Converte as listas para arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Pergunta o color mode
color_mode = input("Enter the color mode (grayscale, rgb, negative, hsv, binary, blur or canny): ").strip().lower()
while color_mode not in ["grayscale", "rgb", "negative", "hsv" , "binary", "blur", "canny"]:
    print("Invalid color mode. Please choose grayscale, rgb, negative, hsv, binary, blur or canny.")
    color_mode = input("Enter the color mode (grayscale, rgb, negative, hsv, binary, blur or canny): ").strip().lower()

# Carrega os dados de treino e teste
x_train, y_train = load_dataset(train_path, labels_train_file, color_mode)
x_test, y_test = load_dataset(test_path, labels_test_file, color_mode)

# Verifica a dimensão
print('X_train:', x_train.shape)
print('y_train:', y_train.shape)
print('X_test:', x_test.shape)
print('y_test:', y_test.shape)

# Converte as labels para one-hot-encoding
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Dimensões das imagem de entrada
img_rows = img_cols = 60
# Ajusta o número de canais de entrada dependendo do color_mode
if color_mode == "grayscale" or color_mode == "negative" or color_mode == "binary" or color_mode == "canny":
    input_shape = (img_rows, img_cols, 1) 
elif color_mode == "rgb"or color_mode == "hsv" or color_mode == "blur":
    input_shape = (img_rows, img_cols, 3)
num_classes = 4

# Reseta os ids das camadas
tf.keras.backend.clear_session()

# Pergunta se quer utilizar um agente pré-treinado anteriormente
use_pretrained = input("Do you want to use a pre-trained model? (yes/no): ").strip().lower()

if use_pretrained == "yes":
    # Carrega o agente pré-treinado se o modelo estiver salvo na pasta de modelos
    model_path = os.path.join(models_folder, f"last_model_{color_mode}.keras")
    if os.path.exists(model_path):
        mlpModel = tf.keras.models.load_model(model_path)
        print("Pre-trained model loaded successfully.")
    else:
        print("No pre-trained model found. Starting training from scratch.")
        use_pretrained = "no"


if use_pretrained != "yes":

    mlpModel = Sequential()
    mlpModel.add(Flatten(input_shape=input_shape))
    mlpModel.add(Dense(128, activation = "relu"))
    mlpModel.add(Dense(64, activation = "relu"))
    mlpModel.add(Dense(num_classes, activation = "sigmoid"))

    mlpModel.summary()

    mlpModel.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# Pergunta se deseja usar early stopping
early = input("Do you want to use a early stopping? (yes/no): ").strip().lower()

if early == "yes":
  
    # Define o callback de Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitora a perda de validação
        patience=10,         # Número de épocas sem melhoria antes de parar
        restore_best_weights=True,  # Restaurar os melhores pesos encontrados
        verbose=1
    )

    history = mlpModel.fit(
        x_train,
        y_train,
        batch_size = 128,
        epochs = 100,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Salva o modelo com early stopping
    last_model_path = os.path.join(models_folder, f"last_model_{color_mode}_early.keras")
    mlpModel.save(last_model_path)
    print(f"Last model saved to {last_model_path}")

    # Caminho para salvar o histórico do modelo atual com early stopping
    history_path = os.path.join(historys_folder, f"history_{color_mode}_early.json")


if early != "yes":

    history = mlpModel.fit(
        x_train,
        y_train,
        batch_size = 128,
        epochs = 100,
        verbose=1,
        validation_split=0.2
    )

    # Salva o modelo sem early stopping
    last_model_path = os.path.join(models_folder, f"last_model_{color_mode}.keras")
    mlpModel.save(last_model_path)
    print(f"Last model saved to {last_model_path}")

    # Caminho para salvar o histórico do modelo atual sem early stopping
    history_path = os.path.join(historys_folder, f"history_{color_mode}.json")

# Salva o histórico
with open(history_path, "w") as f:
    json.dump(history.history, f)

# Plota o gráfico de loss no teste e na validação
plt.figure(figsize=(6, 5))
plt.plot(history.epoch, history.history['loss'], label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

# Salva o plot na pasta de resultados
if early == "yes":
    plot_path = os.path.join(results_folder, f"training_validation_loss_curves_{color_mode}_early.png")
    plt.savefig(plot_path)
    plt.close() 
    print(f"Plots saved to {plot_path}")

if early != "yes":
    plot_path = os.path.join(results_folder, f"training_validation_loss_curves_{color_mode}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plots saved to {plot_path}")

# Plota o gráfico da acurácia no teste e na validação
plt.figure(figsize=(6, 5))
plt.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

# Salva o plot na pasta de resultados
if early == "yes":
    plot_path = os.path.join(results_folder, f"training_validation_accuracy_curves_{color_mode}_early.png")
    plt.savefig(plot_path)
    plt.close()  
    print(f"Plots saved to {plot_path}")

if early != "yes":
    plot_path = os.path.join(results_folder, f"training_validation_accuracy_curves_{color_mode}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plots saved to {plot_path}")

scores = mlpModel.evaluate(
    x_test,
    y_test,
    verbose=1
)

print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]}")

# Faz as previsões no conjunto de teste
y_pred = mlpModel.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1) 

# Cria a matriz de confusão
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Calcula a acurácia
accuracy = accuracy_score(y_true_classes, y_pred_classes)

# Define as labels das classes
class_labels = ["glioma", "saudável", "meningioma", "pituitário"]

# Plota a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão')
plt.text(1.1, -0.1, f'Test Accuracy: {accuracy:.2%}', 
                 horizontalalignment='right', 
                 verticalalignment='top', 
                 transform=plt.gca().transAxes, 
                 fontsize=8, color='black', 
                 bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()

# Salva a matriz de confusão na pasta de resultados
if early == "yes":
    conf_matrix_path = os.path.join(results_folder, f"confusion_matrix_{color_mode}_early.png")
    plt.savefig(conf_matrix_path)

if early != "yes":
    conf_matrix_path = os.path.join(results_folder, f"confusion_matrix_{color_mode}.png")
    plt.savefig(conf_matrix_path)

print(f"Matriz de Confusão salva em: {conf_matrix_path}")
