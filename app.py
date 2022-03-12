from flask import Flask, render_template, request
import cv2
import numpy as np
import base64 
app = Flask(__name__)

def detect_faces(img):
    faces_list = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)  
    if  len(faces) == 0:
        return faces_list
    
    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {}
        face_dict['face'] = gray[y:y + w, x:x + h]
        face_dict['rect'] = faces[i]
        faces_list.append(face_dict)
    return faces_list
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    faces = detect_faces(image)

    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)
        for item in faces:
            draw_rectangle(image, item['rect'])
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodebytes(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)


if __name__ == "__main__":
    app.run(debug=True)