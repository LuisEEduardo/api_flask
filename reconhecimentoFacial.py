import cv2

#cria um identificador EigenFaces
EigenIdent = cv2.face.EigenFaceRecognizer_create()

#Le o arquivo treino
EigenIdent.read("classEigen.yml")

# Utiliza novamente o Haar Cascade para detecção
haarDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#mesmos parâmetros do treino
largura, altura = 220, 220

#Captura da imagem
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    facesDetectadas = haarDetector.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.01,
                                                    minSize=(100,100))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2)
        
        id, nome = EigenIdent.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = 'Carlos'
        elif id == 2:
            nome = 'Augusto'
        elif id == 3:
            nome = 'Pedro'
        elif id == 4:
            nome = 'Sergio'
        elif id == 5:
            nome = 'Carla'
        elif id == 6:
            nome = 'Joana'
        elif id == 7:
            nome = 'Maria'
        elif id == 8:
            nome = 'Raimundo'
        elif id == 9:
            nome = 'Gerson'
            
        cv2.putText(imagem, nome, (x,y +(a+30)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,255,0))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()