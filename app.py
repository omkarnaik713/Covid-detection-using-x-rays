from flask import Flask , request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np 
import io
from PIL import Image 

model = load_model('/Users/omkarnaik/Bankruptcy/.model.h5')

app = Flask(__name__)

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
        
        class_idx = np.argmax(prediction,axis = 1)[0]
        class_label = ['Covid','Normal','Viral Pneumonia']
        predicted_label = class_label[class_idx]
        
        return jsonify({
            'prediction':prediction ,
            'prediction_label':predicted_label
        })
    except Exception as e :
        return jsonify({'error':str(e)}),420

if __name__ == '__main__' :
    app.run(host = '0.0.0.0')