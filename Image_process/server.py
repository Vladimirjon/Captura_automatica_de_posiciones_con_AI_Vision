from flask import Flask, request
import time
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    # Con multipart/form-data Flask debería procesar request.files
    file = request.files.get('image')
    if file:
        filename = f"{int(time.time())}_{file.filename}"
        if not os.path.exists('received_images'):
            os.makedirs('received_images')
        file.save(os.path.join('received_images', filename))
        print("Image saved successfully:", filename)
        return "Success", 200
    else:
        print("No se recibió el archivo en request.files")
        return "Failed", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
