import cv2

def reconhecimento(imagemPath):

    #cria um identificador EigenFaces
    EigenIdent = cv2.face.EigenFaceRecognizer_create()

    #Le o arquivo treino
    EigenIdent.read("classEigen.yml")

    # Utiliza novamente o Haar Cascade para detecção
    haarDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # mesmos parâmetros do treino
    largura, altura = 220, 200

    while (True):        
        # Abrindo a imagem
        imagem = cv2.imread(imagemPath)        

        # Redimensionando o tamanho da imagem
        resizeImage = cv2.resize(imagem, (220, 200), interpolation = cv2.INTER_NEAREST)

        # Transformando a imagem para cinza    
        imagemCinza = cv2.cvtColor(resizeImage, cv2.COLOR_BGR2GRAY)
        
        # Deteccao
        facesDetectadas = haarDetector.detectMultiScale(imagemCinza, scaleFactor=1.01, minSize=(100,100))
        
        id = 0


        for (x, y, l, a) in facesDetectadas:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))

            id, acuracia = EigenIdent.predict(imagemFace)

        return id, acuracia
