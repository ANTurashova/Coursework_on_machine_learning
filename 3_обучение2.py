import keras
import cv2
import os
import numpy as np
import random

numberPeople = 5
epochsLear = 10
VideoPatch = "video/"

def createNeuralNetwork(inputShape):
    """"Создаём модель нейросети"""
    neuralNetwork = keras.models.Sequential()
    neuralNetwork.add(keras.layers.Conv2D(48, kernel_size=11, activation='relu', inputShape=inputShape))
    neuralNetwork.add(keras.layers.Conv2D(128, kernel_size=5, activation='relu'))
    neuralNetwork.add(keras.layers.MaxPooling2D(pool_size=3))
    neuralNetwork.add(keras.layers.Dropout(.25))
    neuralNetwork.add(keras.layers.Conv2D(192, kernel_size=3, activation='relu'))
    neuralNetwork.add(keras.layers.MaxPooling2D(pool_size=3))
    neuralNetwork.add(keras.layers.Dropout(.25))
    neuralNetwork.add(keras.layers.Conv2D(192, kernel_size=3, activation='relu'))
    neuralNetwork.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    neuralNetwork.add(keras.layers.Flatten())
    neuralNetwork.add(keras.layers.Dense(2048, activation='relu'))
    neuralNetwork.add(keras.layers.Dense(2048, activation='relu'))
    neuralNetwork.add(keras.layers.Dense(500, activation='relu'))
    return neuralNetwork

def readDataSet():
    """Получаем список с преобразованными пикселями фотографии"""
    indexNumber = 0
    indexFoto = []
    listFoto = []
    listDir = os.listdir(VideoPatch)
    for i in listDir:
        if os.path.isdir(VideoPatch + i):
            next = os.listdir(VideoPatch + i)
            for a in next:
                imageAddress = cv2.imread(VideoPatch + i + '/' + a)
                listFoto.append(imageAddress)
                indexFoto.append(indexNumber)
            indexNumber += 1
    indexFoto = np.array(indexFoto)
    listFoto = np.array(listFoto)
    listFoto = np.resize(listFoto, (len(listFoto), 100, 100, 1))
    return listFoto, indexFoto

def createPairsOfPhotos(x, digitIndices):
    """Создаём позитивные и негативные пары"""
    listPairs = []
    listLabels = []
    length = len(digitIndices[0]) - 1
    for i in range(len(digitIndices) - 1):
        if (len(digitIndices[i]) < len(digitIndices[i+1])) and (len(digitIndices[i]) > 1):
            length = len(digitIndices[i]) - 1
        else:
            length = len(digitIndices[i+1]) - 1
    for j in range(numberPeople):
        for i in range(length):
            a1, a2 = digitIndices[j][i], digitIndices[j][i + 1]
            listPairs += [[x[a1], x[a2]]]
            rand = random.randrange(1, numberPeople)
            jrand = (j + rand) % numberPeople
            a1, a2 = digitIndices[j][i], digitIndices[jrand][i]
            listPairs += [[x[a1], x[a2]]]
            listLabels += [1, 0]
    return np.array(listPairs), np.array(listLabels)

def energyFunction(vectors):
    """Функция расчета энергии, евклидово расстояние"""
    x, y = vectors
    sumSquare = keras.backend.sum(keras.backend.square(x - y), axis=1, keepdims=True)
    return keras.backend.sqrt(keras.backend.maximum(sumSquare, keras.backend.epsilon()))

def outputShapeEnergyFunction(inletShapes):
    """Функция для вывода расчета энергии"""
    inletShapes1, inletShapes2 = inletShapes
    return (inletShapes1[0], 1)

def lossContrastive(yTrue, yPrediction):
    """Функция контрастной потери"""
    profit = 1
    square = keras.backend.square(yPrediction)
    profitSquare = keras.backend.square(keras.backend.maximum(profit - yPrediction, 0))
    return keras.backend.mean(yTrue * square + (1 - yTrue) * profitSquare)

def modelPerformanceEvaluation(yTrue, yPrediction):
    """Функция метрики, используется для оценки производительности модели"""
    prediction = yPrediction.ravel() < 0.5
    return np.mean(prediction == yTrue)

def classificationAccuracy(yTrue, yPrediction):
    """Функция, вычисляющая точность классификации с фиксированным порогом расстояний"""
    return keras.backend.mean(keras.backend.equal(yTrue, keras.backend.cast(yPrediction < 0.5, yTrue.dtype)))

xTrain, yTrain = readDataSet()
xTrain = xTrain.astype('float32')
xTrain /= 255
inputShape = xTrain.shape[1:]
digitIndices = [np.where(yTrain == i)[0] for i in range(numberPeople)]
trainPairs, trainY = createPairsOfPhotos(xTrain, digitIndices)
testPairs = trainPairs[0:200]
testY = trainY[0:200]
trainPairs = trainPairs[200::]
trainY = trainY[200::]

baseNetwork = createNeuralNetwork(inputShape)
input1 = keras.layers.Input(shape=inputShape)
input2 = keras.layers.Input(shape=inputShape)
processed1 = baseNetwork(input1)
processed2 = baseNetwork(input2)
distance = keras.layers.Lambda(energyFunction, output_shape=outputShapeEnergyFunction)([processed1, processed2])
model = keras.models.Model([input1, input2], distance)

""" Запускаем обучение нейронной сети:"""
rmsprop = keras.optimizers.RMSprop()
model.compile(loss=lossContrastive, optimizer=rmsprop, metrics=[classificationAccuracy])
history = model.fit([trainPairs[:, 0], trainPairs[:, 1]], trainY, batch_size=30, epochsLear=1, validation_data=([testPairs[:, 0], testPairs[:, 1]], testY), verbose=2)
model.save_weights('checkpoint')
yPrediction = model.predict([trainPairs[:, 0], trainPairs[:, 1]])
trainAcc = modelPerformanceEvaluation(trainY, yPrediction)
yPrediction = model.predict([testPairs[:, 0], testPairs[:, 1]])
testAcc = modelPerformanceEvaluation(testY, yPrediction)
print('* Точность на тренировочной выборке: %0.2f%%' % (100 * trainAcc))
print('* Точность на испытательном наборе: %0.2f%%' % (100 * testAcc))








