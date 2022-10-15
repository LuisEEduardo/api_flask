from flask import Flask, jsonify, request
from processaImagem import processamento
from treino import realiza_treino
from reconhecimentoFacial import reconhecimento
import os

app = Flask("aps")

app.config["IMAGE_UPLOADS"] = "C:\\dev\\faculdade\\aps_6_semestre\\api_flask\\tmp"

@app.route("/", methods=["GET"])
def index():
    return jsonify("Api de reconhecimento facial")


@app.route("/processamentoImagem", methods=["GET"])
def processamento_imagem():
    processamento()
    return jsonify("ok")


@app.route("/treino", methods=["GET"])
def treino():
    realiza_treino()
    return jsonify("ok")


@app.route("/reconhecimento", methods=["POST"])
def reconhecimento_facial():
    imagem = request.files["image"]    

    pathArquivo = os.path.join(app.config["IMAGE_UPLOADS"], imagem.filename)

    imagem.save(pathArquivo)

    id = reconhecimento(pathArquivo)

    os.remove(pathArquivo)

    return jsonify(id)


app.run()
