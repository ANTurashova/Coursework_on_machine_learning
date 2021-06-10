import keras
import cv2
import numpy as np

VideoPatch = 'video_sources/1.mp4'
Play = True
cv2font = cv2.FONT_HERSHEY_SIMPLEX
cascadeFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascadeEye = cv2.CascadeClassifier('haarcascade_eye.xml')

def giveOutSimilarityRate(videoImage):
    """Функция сверяет загруженные изобажения с лицами людей в кадре и выдаёт коэффициент сходства"""
    result1 = []
    Oleg = np.zeros([1, 2, 100, 100])
    Sergey = np.zeros([1, 2, 100, 100])
    tmp1 = cv2.imread('s/s_Oleg/2_426.jpg', cv2.IMREAD_UNCHANGED)
    tmp2 = cv2.imread('s/s_Sergey/2_134.jpg', cv2.IMREAD_UNCHANGED)
    Oleg[0, 0, :, :] = tmp1
    Oleg[0, 1, :, :] = videoImage
    Oleg /= 255
    Oleg = Oleg.reshape(1, 2, 100, 100, 1)
    Sergey[0, 0, :, :] = tmp2
    Sergey[0, 0, :, :] = videoImage
    Sergey /= 255
    Sergey = Sergey.reshape(1, 2, 100, 100, 1)
    yPrediction = model.predict([Oleg[:, 0], Oleg[:, 1]])
    result1.append(yPrediction[0])
    yPrediction = model.predict([Sergey[:, 0], Sergey[:, 1]])
    result1.append(yPrediction[0])
    names = ['Oleg', 'Sergey']
    result1 = np.array(result1)
    return names[np.argmin(result1)], result1[np.argmin(result1)]

def createNeuralNetwork(inputShape): #+
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

def energyFunction(vectors): #+
    """Функция расчета энергии, евклидово расстояние"""
    x, y = vectors
    sumSquare = keras.backend.sum(keras.backend.square(x - y), axis=1, keepdims=True)
    return keras.backend.sqrt(keras.backend.maximum(sumSquare, keras.backend.epsilon()))

def outputShapeEnergyFunction(inletShapes): #+
    """Функция для вывода расчета энергии"""
    inletShapes1, inletShapes2 = inletShapes
    return (inletShapes1[0], 1)

def lossContrastive(yTrue, yPrediction): #+
    """Функция контрастной потери"""
    profit = 1
    square = keras.backend.square(yPrediction)
    profitSquare = keras.backend.square(keras.backend.maximum(profit - yPrediction, 0))
    return keras.backend.mean(yTrue * square + (1 - yTrue) * profitSquare)

inputShape = [100, 100, 1]
baseNetwork = createNeuralNetwork(inputShape)
input1 = keras.layers.Input(shape=inputShape)
input2 = keras.layers.Input(shape=inputShape)
processed1 = baseNetwork(input1)
processed2 = baseNetwork(input2)
distance = keras.layers.Lambda(energyFunction, output_shape=outputShapeEnergyFunction)([processed1, processed2])
model = keras.models.Model([input1, input2], distance)
print(model.summary())
rmsprop = keras.optimizers.RMSprop()
model.compile(loss=lossContrastive, optimizer=rmsprop)
model.load_weights('checkpoint')
videoCap = cv2.VideoCapture(VideoPatch)

while (videoCap.isOpened()) and Play:
    v, vFrame = videoCap.read()
    grayColor = cv2.cvtColor(vFrame, cv2.COLOR_BGR2GRAY)
    foundFaces = cascadeFace.detectMultiScale(grayColor, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in foundFaces:
        grayRoi = grayColor[y:y + h, x:x + w]
        foundEyes = cascadeEye.detectMultiScale(grayRoi, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        if len(foundEyes) > 0:
            cv2.rectangle(vFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            size = (100, 100)
            out = cv2.resize(grayRoi, size, interpolation=cv2.INTER_AREA)
            text, ping = giveOutSimilarityRate(out)
            text = text + ' ' + str(ping)
            cv2.putText(vFrame, text, (x + 5, y - 5), cv2font, 0.5, (255, 255, 255), 1)
    if v == True: cv2.imshow('Frame', vFrame)
    else: break
    if cv2.waitKey(10) == 27: Play = False
    
videoCap.release()
cv2.destroyAllWindows()
