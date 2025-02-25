import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuração de Log
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MLP:
    meanSqErrorEpoch = []
    bias = 1
    np.random.seed(seed = 404)

    # Construtor de inicialização
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.Wh = np.random.uniform(-0.5, 0.5, (hidden, inputs + self.bias)) 
        self.Wo = np.random.uniform(-0.5, 0.5, (outputs, hidden + self.bias)) 

    # Ativação sigmoidal
    def sigmoideActivation(self, net):
        return 1 / (1 + np.exp(-net))

    # Função que realiza a ativação sigmoidal derivativa
    def derivativeSigmoideActivation(self, output):
        return output * (1 - output)

    # Função que realiza o forward
    def forward(self, inpVector):
        exemplo_bias = np.append(inpVector, 1)  
        NetHidden = np.dot(self.Wh, exemplo_bias)
        Inp_Hidden = self.sigmoideActivation(NetHidden)

        Inp_Hidden_bias = np.append(Inp_Hidden, 1)  
        NetOutputs = np.dot(self.Wo, Inp_Hidden_bias)
        O_outputs = self.sigmoideActivation(NetOutputs)

        return O_outputs, NetHidden, NetOutputs, Inp_Hidden

    # Função de treino do agente
    def train(self, dataset, iterations, learnRate, threshold):
        for epoch in range(iterations):
            correct_predictions = 0
            sqError = 0

            for inpVector, classe in dataset:
                O_outputs, NetHidden, NetOutputs, Inp_Hidden = self.forward(inpVector)
                error = classe - O_outputs
                # Começa o backward
                deltaO = error * self.derivativeSigmoideActivation(O_outputs)
                Inp_Hidden_bias = np.append(self.sigmoideActivation(NetHidden), 1)
                deltaH = self.derivativeSigmoideActivation(Inp_Hidden) * np.dot(self.Wo.T, deltaO)[:-1]
                self.Wo += learnRate * np.outer(deltaO, Inp_Hidden_bias)
                self.Wh += learnRate * np.outer(deltaH, np.append(inpVector, 1))
                sqError += np.sum(error ** 2)
                # Contar acertos
                predicted_class = np.argmax(O_outputs)
                real_class = np.argmax(classe)
                if predicted_class == real_class:
                    correct_predictions += 1

            sqError /= len(dataset)
            self.meanSqErrorEpoch.append(sqError)
            accuracy = correct_predictions / len(dataset)
            logging.info(f'Epoch {epoch+1}/{iterations} -> MSE: {sqError:.3f}, Accuracy: {accuracy:.3f}')

            # Parada
            if sqError < threshold:
                break

        return self.meanSqErrorEpoch, accuracy
    
    # Função para testar o agente
    def test(self, dataset):
        correct_predictions = 0
        sqError = 0  

        for inpVector, classe in dataset:
            O_outputs, _, _, _ = self.forward(inpVector)
            error = classe - O_outputs
            sqError += np.sum(error ** 2)

            predicted_class = np.argmax(O_outputs)
            real_class = np.argmax(classe)
            if predicted_class == real_class:
                correct_predictions += 1

        sqError /= len(dataset)
        accuracy = correct_predictions / len(dataset)
        return sqError, accuracy

    # Plot gráfico do erro quadrático médio por época
    def plotError(self, results_folder):
        plt.figure(figsize=(10, 6))
        plt.plot(self.meanSqErrorEpoch, marker='o', linestyle='-', color='black', markersize=2, linewidth=1)
    
        plt.title('Mean Squared Error per Epoch', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Mean Squared Error', fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Exibir a média e desvio padrão 
        mean_mse = np.mean(self.meanSqErrorEpoch)
        plt.axhline(mean_mse, color='red', linestyle='--', label='Standard Deviation')
        
        plt.legend()
        plt.tight_layout()

        # Salva o gráfico na pasta de resultados
        plot_path = os.path.join(results_folder, f"mean_square_error.png")
        plt.savefig(plot_path)
        plt.close() 
        print(f"Plots saved to {plot_path}")         

    # Plot gráfico da matriz de confusão
    def plotConfusionMatrix(self, dataset, class_labels, results_folder):
        predictions = []
        ground_truth = []

        # Preve classes para o dataset
        for inpVector, classe in dataset:
            O_outputs, _, _, _ = self.forward(inpVector)
            predicted_class = np.argmax(O_outputs)
            real_class = np.argmax(classe)
            predictions.append(predicted_class)
            ground_truth.append(real_class)

        # Gera a matriz de confusão
        cMatrix = confusion_matrix(ground_truth, predictions)

        # Plota a matriz de confusão
        disp = ConfusionMatrixDisplay(confusion_matrix=cMatrix, display_labels=class_labels)
        disp.plot(cmap='Greens', values_format='d')

        # Legenda com o percentual de acerto
        accuracy = np.trace(cMatrix) / np.sum(cMatrix)
        plt.title('Confusion Matrix')
        plt.text(1.1, -0.1, f'Test Accuracy: {accuracy:.2%}', 
                 horizontalalignment='right', 
                 verticalalignment='top', 
                 transform=plt.gca().transAxes, 
                 fontsize=8, color='black', 
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.tight_layout()

        # Salva a matriz de confusão na pasta de resultados
        confusion_matrix_path = os.path.join(results_folder, f"confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"Matriz de confusão salva em {confusion_matrix_path}")
