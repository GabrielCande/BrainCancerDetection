import matplotlib.pyplot as plt
import os
import json

# Pasta onde os resultados estão armazenados
historys_folder = "BrainCancerDetection/Historicos"
results_folder = "Graficos"

# Dicionário com os caminhos dos históricos de cada modelo
history_files = {
    "DNN": os.path.join(historys_folder, "DNN", "history_grayscale_early.json"),
    "CNN": os.path.join(historys_folder, "CNN", "history_hsv_early.json"),
    "KMLP": os.path.join(historys_folder, "KMLP", "history_hsv_early.json"),
    "VGG16": os.path.join(historys_folder, "VGG16", "history_negative.json"),
    "ResNet50": os.path.join(historys_folder, "ResNet50", "history_grayscale.json"),
    "InceptionV3": os.path.join(historys_folder, "InceptionV3", "history_negative.json"),
}

# Cores para cada modelo
model_colors = {
    "DNN": "blue",
    "CNN": "red",
    "KMLP": "purple",
    "VGG16": "gray",
    "ResNet50": "green",
    "InceptionV3": "yellow"
}

# Dicionários para armazenar os históricos
histories = {}

# Carrega os históricos salvos
for model_name, file_path in history_files.items():
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            histories[model_name] = json.load(f)

# Cria o gráfico comparando o loss de treino
plt.figure(figsize=(6, 5))
for model_name, history in histories.items():
    plt.plot(history["loss"], label=f"{model_name}", color=model_colors[model_name])

plt.title("Loss do Treino")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(results_folder, "training_loss.png"))
plt.close()

# Cria o gráfico comparando o loss de validação
plt.figure(figsize=(6, 5))
for model_name, history in histories.items():
    plt.plot(history["val_loss"], label=f"{model_name}", color=model_colors[model_name])

plt.title("Loss da Validação")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(results_folder, "validation_loss.png"))
plt.close()

# Cria o gráfico comparando a acurácia de treino
plt.figure(figsize=(6, 5))
for model_name, history in histories.items():
    plt.plot(history["accuracy"], label=f"{model_name}", color=model_colors[model_name])

plt.title("Acurácia do Treino")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.savefig(os.path.join(results_folder, "training_accuracy.png"))
plt.close()

# Cria o gráfico comparando a acurácia de validação
plt.figure(figsize=(6, 5))
for model_name, history in histories.items():
    plt.plot(history["val_accuracy"], label=f"{model_name}", color=model_colors[model_name])

plt.title("Acurácia da Validação")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.savefig(os.path.join(results_folder, "validation_accuracy.png"))
plt.close()

print("Gráficos comparativos salvos em 'Resultados_Experimentos'")