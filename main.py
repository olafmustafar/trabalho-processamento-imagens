import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
from progress.bar import Bar
import time
from datetime import datetime

SVC_C=1
SVC_GAMMA=0.0001
HIST_SIZE = 20
HIST_RANGE = (10,246)

TRAIN_INPUT='fruits-360/Training/'
TEST_INPUT='fruits-360/Test/'
FRUITS_TO_TRAIN = ["Kiwi",
                   "Pepper Green",
                   "Watermelon",
                   "Raspberry",
                   "Potato White",
                   "Apple Red 1",
                   "Pineapple",
                   "Avocado",
                   "Carambula" ]

def main():
    #Um scaler sera utilizado para normalizar os dados
    scaler = StandardScaler()

    print("-> Fase de Treinamento")
    trainPath = TRAIN_INPUT
    trainImages, trainLabels = getData(trainPath)
    trainLabels, encoder = encodeLabels(trainLabels)
    trainData = processImages(trainImages);
    trainData = scaler.fit_transform(trainData)

    #Realizando o treinamento utilizando o algoritmo SVC
    classifier = SVC(C=SVC_C, gamma=SVC_GAMMA)
    classifier.fit(trainData, trainLabels)

    print("-> Fase de Teste")
    testPath = TEST_INPUT
    testImages, testLabels = getData(testPath)
    testLabels, _ = encodeLabels(testLabels)
    testData = processImages(testImages);
    testData = scaler.transform(testData)

    #Realizando a predição sobre as imagens de teste
    predictedLabels = classifier.predict(testData)

    #Mostrando os resultados
    accuracy = plotConfusionMatrix(encoder,testLabels,predictedLabels)

def predict_single(img):
    # Método util para testar uma imagem
    scaler = StandardScaler()
    trainPath = TRAIN_INPUT
    trainImages, trainLabels = getData(trainPath)
    trainLabels, encoder = encodeLabels(trainLabels)
    trainData = processImages(trainImages);
    trainData = scaler.fit_transform(trainData)
    classifier = SVC(C=SVC_C, gamma=SVC_GAMMA)
    classifier.fit(trainData, trainLabels)
    data = processImages([img]);
    data = scaler.transform(data)
    result = classifier.predict(data)
    return encoder.inverse_transform(result)

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        folderList = os.listdir(path)
        bar = Bar('-> Buscando imagens e labels...',max=len(FRUITS_TO_TRAIN),suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
        for folder in folderList:
            label = os.path.basename(folder);
            if label not in FRUITS_TO_TRAIN:
                continue

            fileList = os.listdir(path+folder)
            for file in fileList:
                image = cv2.imread(path+ folder+'/'+file);
                images.append(image)
                labels.append(label)

            bar.next()
        bar.finish()
    return images, labels

def encodeLabels(labels):
    print("-> Processando labels...")
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return labels, encoder

def processImages(images):
    processed = []
    bar = Bar('-> Processando imagens...',max=len(images),suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
    for img in images:
        resized = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)

        # Criando um histgrama para cada cor (azul, verde, vermelho)
        bgr = cv2.split(resized)
        b_hist = cv2.calcHist(bgr, [0], None, [HIST_SIZE], HIST_RANGE, accumulate=False)
        g_hist = cv2.calcHist(bgr, [1], None, [HIST_SIZE], HIST_RANGE, accumulate=False)
        r_hist = cv2.calcHist(bgr, [2], None, [HIST_SIZE], HIST_RANGE, accumulate=False)

        # Concatenando e normalizando os histogramas
        hist = np.array([r_hist,g_hist,b_hist]).flatten()
        cv2.normalize(hist, hist)

        # Vetor da imagem (em escala de cinza)
        flatten = resized.flatten().astype(np.float32)/255

        # Adicionando o vetor da imagem + histograma para os dados de treinamento
        processed.append(np.concatenate(( hist,flatten )))
        bar.next()
    bar.finish()
    return processed

def plotConfusionMatrix(testEncoder,testLabels,predictedLabels):
    test = testEncoder.inverse_transform(testLabels)
    pred = testEncoder.inverse_transform(predictedLabels)
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test,pred,ax=ax, colorbar=False, cmap=plt.cm.Greens)
    plt.suptitle('Confusion Matrix: ',fontsize=18)
    accuracy = metrics.accuracy_score(testLabels,predictedLabels)*100
    plt.title(f'Accuracy: {accuracy}%',fontsize=18,weight='bold')
    now = datetime.now().strftime('%d%m%Y-%H%M')
    plt.savefig(f'./results/{now}', dpi=300)  
    plt.show()
    return accuracy

if __name__ == "__main__":
    main()
