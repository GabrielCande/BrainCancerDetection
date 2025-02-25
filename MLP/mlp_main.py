import os
import cv2
import logging
import numpy as np

from mlp import MLP
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------------------------
# Configuração de Log
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------------------------------------
# Cria uma pasta para salvar os resultados gráficos
results_folder = "MLPresults"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Função para carregar as imagens e as labels
def load_dataset(image_dir, label_file, img_size=(60, 60)):
    images = []
    labels = []

    # Carrega as labels do arquivo
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Carrega o nome da imagem e a label referente na linha do arquivo de labels
        img_name, label = line.strip().split()
        # Carrega as imagens de acordo com o nome que foi pego no arquivo de labels
        img_path = os.path.join(image_dir, img_name)

        # Carrega a imagem com o filtro grayscale, reescala e normaliza
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = img.astype("float32") / 255.0
        img = img.reshape(img_size[0], img_size[1])

        images.append(img)
        labels.append(int(label))

    # Converte as listas para arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# -------------------------------------------------------------------
# Caminhos das pastas e arquivos de labels
train_path = "BrainCancerDetection/dataset/X_train"
test_path = "BrainCancerDetection/dataset/X_test"
labels_train_file = "BrainCancerDetection/dataset/y_train/y_train.txt"
labels_test_file = "BrainCancerDetection/dataset/y_test/y_test.txt"

# Carrega o dataset com o filtro grayscale
x_train, y_train = load_dataset(train_path, labels_train_file)
x_test, y_test = load_dataset(test_path, labels_test_file)

# Redimensiona as imagens para vetores de 3600 elementos (60x60)
X_train = x_train.reshape(x_train.shape[0], 3600)
X_test = x_test.reshape(x_test.shape[0], 3600)

# Normalizar os dados (garantir que estejam entre 0 e 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converte as labels para one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)

# -------------------------------------------------------------------
# Configurações da MLP
iterations = 100
learnRate = 0.1
threshold = 0.0001

# Instancia a MLP:
mlp = MLP(inputs=3600, hidden=128, outputs=4)

# Treinamento
logging.info('Initializing MLP Agent training.')
train_mse, train_accuracy = mlp.train(list(zip(X_train, y_train_one_hot)), iterations, learnRate, threshold)

# Teste
logging.info('Initializing MLP Agent test.')
test_mse, test_accuracy = mlp.test(list(zip(X_test, y_test_one_hot)))

# Exibe os resultados
logging.info('Execution complete.')
print("-------------------------------------------------------------------------")
print("Final values:")
print(f"Train -> MSE: {train_mse[-1]:.3f}, Accuracy: {train_accuracy:.3f}")
print(f"Test -> MSE: {test_mse:.3f}, Accuracy: {test_accuracy:.3f}")
print("-------------------------------------------------------------------------")

# Plota o gráfico do erro quadrático médio por época
mlp.plotError(results_folder)

# Plota a matriz de confusão
class_labels = [str(i) for i in range(4)]
mlp.plotConfusionMatrix(list(zip(X_test, y_test_one_hot)), class_labels, results_folder)