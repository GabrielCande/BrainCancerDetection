import os
import shutil
import random

# Define os caminhos do dataset
dataset_path = "BrainCancerDetection/imagens"
train_path = "BrainCancerDetection/dataset/train"
test_path = "BrainCancerDetection/dataset/test"


# Cria as pastas de treino e teste
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Quantidade fixa de imagens para treino
train_size = 1400  

# Lista todas as classes (pastas)
classes = os.listdir(dataset_path)

for class_name in classes:
    # Carrega a pasta por classe
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Lista todas as imagens da classe
    images = os.listdir(class_dir)
    
    # Embaralha a lista de imagens para evitar viés
    random.shuffle(images)

    # Define treino e teste
    train_images = images[:train_size] 
    test_images = images[train_size:] 

    # Cria as pastas para a classe dentro das pastas train e test
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    # Move as imagens para treino
    for img in train_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(train_path, class_name, img))

    # Move as imagens para teste
    for img in test_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(test_path, class_name, img))

print("✅ Separação concluída!")

# Caminhos das pastas X_train e y_train
mixed_train_path = "BrainCancerDetection/dataset/X_train"
labels_train_path = "BrainCancerDetection/dataset/y_train"
labels_file = os.path.join(labels_train_path, "y_train.txt")

# Cria as pastas para as imagens e as labels de treino
os.makedirs(mixed_train_path, exist_ok=True)
os.makedirs(labels_train_path, exist_ok=True)

# Define as classes numericamente
class_mapping = {}
all_images = []

# Lista todas as classes (pastas) dentro da pasta train, ordena alfabéticamente (glioma, healthy, meningioma e pituitary) e atribui rótulos de 0 a 3
classes = sorted(os.listdir(train_path))
for i, class_name in enumerate(classes):
    class_mapping[class_name] = i
    class_dir = os.path.join(train_path, class_name)

    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        for img in images:
            img_path = os.path.join(class_dir, img)
            # Salva o caminho + classe numérica
            all_images.append((img_path, i))

# Mistura as imagens aleatoriamente para evitar viés
random.shuffle(all_images)

# Cria o arquivo de labels para o treino
with open(labels_file, "w") as f:
    for i, (img_path, label) in enumerate(all_images):
        # Renomeia a imagem
        new_img_name = f"image_{i}.jpg"
        new_img_path = os.path.join(mixed_train_path, new_img_name)

        # Move a imagem para a nova pasta X_train
        shutil.move(img_path, new_img_path)

        # Escreve o nome da imagem junto com a label correspondente no arquivo de texto
        f.write(f"{new_img_name} {label}\n")

print("✅ Imagens e labels de treino salvas!")

# Caminhos das pastas X_test e y_test
mixed_test_path = "BrainCancerDetection/dataset/X_test"
labels_test_path = "BrainCancerDetection/dataset/y_test"
labels_file = os.path.join(labels_test_path, "y_test.txt")

# Cria as pastas para as imagens e as labels de teste
os.makedirs(mixed_test_path, exist_ok=True)
os.makedirs(labels_test_path, exist_ok=True)

# Define as classes numericamente
class_mapping = {}
all_images = []

# Lista todas as classes (pastas) dentro a pasta test, ordena alfabéticamente (glioma, healthy, meningioma e pituitary) e atribui rótulos de 0 a 3
classes = sorted(os.listdir(test_path))
for i, class_name in enumerate(classes):
    class_mapping[class_name] = i
    class_dir = os.path.join(test_path, class_name)

    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        for img in images:
            img_path = os.path.join(class_dir, img)
            # Salva o caminho + classe numérica
            all_images.append((img_path, i))

# Mistura as imagens aleatoriamente para evitar viés
random.shuffle(all_images)

# Cria o arquivo de labels para o teste
with open(labels_file, "w") as f:
    for i, (img_path, label) in enumerate(all_images):
        # Renomeia a imagem
        new_img_name = f"test_image_{i}.jpg"
        new_img_path = os.path.join(mixed_test_path, new_img_name)

        # Move a imagem para a nova pasta X_test
        shutil.move(img_path, new_img_path)

        # Escreve o nome da imagem junto com a label correspondente no arquivo de texto
        f.write(f"{new_img_name} {label}\n")

print("✅ Imagens e labels de teste salvas!")

