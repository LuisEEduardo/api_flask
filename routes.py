from flask import Flask, jsonify
from processaImagem import processamento
from treino import realiza_treino

app = Flask("aps")


@app.route("/", methods=["GET"])
def index():
    return jsonify("teste")


@app.route("/processamentoImagem", methods=["GET"])
def processamento_imagem():
    processamento()
    return jsonify("ok")


@app.route("/treino", methods=["GET"])
def treino():
    realiza_treino()
    return jsonify("ok")



app.run()
