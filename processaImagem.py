import cv2
import os

def processamento():
    path = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    largura, altura = 220, 220
    imagemOriginal = 0

    for pessoa in path:
        print("Processamento em andamento: imagem["+ pessoa +"]")
        imagemOriginal = cv2.imread(pessoa)
        
        #transformar a imagem para escala de cinza
        imagemCinza = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza,scaleFactor = 1.01, minSize = (30,30))
        
        # Retângulo na face detectada
        for (x,y,l,a) in facesDetectadas:
            cv2.rectangle(imagemOriginal, (x,y), (x + l, y + a),(0,0,255),2)
            
            #Redimensionar a foto
            imagemFinal = cv2.resize(imagemCinza[y: y+ a, x:x+l], (largura, altura))
        
            #equalizar a imagem
            ImageFaceEqual = imgEqualist = cv2.equalizeHist(imagemFinal)
            #print("\/../detectadas/\\" + pessoa)
            #cv2.imwrite("detectadas\\/" + pessoa, ImageFaceEqual)
            cv2.imwrite(pessoa, ImageFaceEqual)

    print("Processamento concluído com sucesso!")