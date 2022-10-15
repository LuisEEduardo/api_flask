import cv2
import os
import numpy as np

def getImagemComId():

    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
   
    for caminho in caminhos:
        imagemOriginal = cv2.imread(caminho)

        resizeImage = cv2.resize(imagemOriginal, (220, 200), interpolation = cv2.INTER_NEAREST)

        imagemCinza = cv2.cvtColor(resizeImage, cv2.COLOR_BGR2GRAY)        
        path = caminho[13:]
        #print(path[0])
        id = int(path[0])
        ids.append(id)

        # resizedGrayImage = cv2.resize(imagemCinza, (780, 540), interpolation = cv2.INTER_NEAREST)

        faces.append(resizeImage)
        
    return np.array(ids), faces   


def realiza_treino():
    eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
    print("Realizando treino")
    ids, faces = getImagemComId()
    #print(faces)
        
    eigenface.train(faces, ids)
    eigenface.write("classEigen.yml")
    print("Treino finalizado.")

