from flask import Flask , request,render_template, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np 
import pickle
from PIL import Image 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping', methods = ['HEAD', 'GET'])
def ping():
    return 200  

def process_image(image) :
    if image.mode != 'RGB' :
        image = image.convert('RGB')
    image = image.resize((128,128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image/255.0
    
    return image 

@app.route("/predict",methods = ['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error":"No file part"}),400
    file = request.files['file']
    
    if file.filename == '' :
        return jsonify({'error':'No selected file'}),410
    
    try :
        image = Image.open(file)
        processed_image = process_image(image)
        prediction = model.predict(processed_image).tolist()
        percent = round(max(prediction[0]) * 100)
        class_idx = np.argmax(prediction,axis = 1)[0]
        class_label = ['Covid','Normal','Viral Pneumonia']
        predicted_label = class_label[class_idx]
        
        if predicted_label == 'Covid':
            return jsonify(f'You have {percent}% chance of having Covid ')
        elif predicted_label == 'Normal' :
            return jsonify('Your  Chest X-ray is Normal')
        else :
            return jsonify(f'You have {percent}% chance of having Viral Pneumonia ')
    except Exception as e :
        return f'Error : {str(e)}'

if __name__ == '__main__' :
    app.run(host = '0.0.0.0',port = 8080)