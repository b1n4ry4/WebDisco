from flask import Flask #import flask class

UPLOAD_FOLDER = 'C:/Users/b1n4ry4/Desktop/merged/static' #upload path

app = Flask(__name__) #__name__ ker gre za 1 modul, za več modulov __main__
app.secret_key = "secret key" #za varnost odjemalca, brez varnosti flask ne omogoča dostopa
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER #določimo path za upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # max 16mb