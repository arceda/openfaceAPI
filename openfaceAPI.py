from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
import cv2
import openface
import numpy as np

app = Flask(__name__)


align = openface.AlignDlib("/root/openface/models/shape_predictor_68_face_landmarks.dat")
net = openface.TorchNeuralNet("/root/openface/models/openface/nn4.small2.v1.t7", 96)

@app.route('/openfaceAPI', methods=['POST'])
@cross_origin()
def openfaceAPI(): 
    if request.method == 'POST':
        f = request.files['file']
        #filename = secure_filename(f.filename)
        filename = f.filename
        f.save("/root/photos/" + filename)       
        #f.save("/home/vicente/Downloads" + filename)       

        img = cv2.imread("/root/photos/" + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bb = align.getLargestFaceBoundingBox(img) # deteccion de rostros
        alignedFace = align.align(96, img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        rep = net.forward(alignedFace) # obtenemos el vector de caracteristicas
        print(rep)

        return "OK"

if __name__ == "__main__":
    app.run(debug=True, port=5000)