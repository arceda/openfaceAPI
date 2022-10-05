from flask import Flask
from flask import request
from flask import jsonify
import requests

app = Flask(__name__) #instancia

@app.route('/openface', methods=['POST']) #wrap (decorator)
def openface(): 
    if request.method == 'POST':
        f = request.files['file']
        
        # si queremos guardar la foto
        filename = f.filename
        path = "D://SOFTWARE_CONSTRUCTION//images//" + filename
        f.save(path)       
           
   
        # call openfaceAPI ##################################
        url = 'http://127.0.0.1:81/openfaceAPI'
        files = {'file': open(path, 'rb')}
        #files = {'file': f}

        result = requests.post(url, files=files)
        print(result.json())
        ######################################################

        # queda pendiente: registrar los demas datas del 
        # usuario en la BD junto con el vector de caracteristicas

    return "OK!"

if __name__ == '__main__':
    app.run(debug=True, port=5000) #lunch server on port 5000