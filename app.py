from keras.models import load_model
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
from pygame import mixer
import os
mixer.init()
sound = mixer.Sound(r"C:\Users\Dell\Desktop\Shojeb ML Projects\CNN5\beep-02.wav")
model = load_model(r"C:\Users\Dell\Desktop\Shojeb ML Projects\CNN5\model-016.model")
sound1 = mixer.Sound(r"C:\Users\Dell\Desktop\Shojeb ML Projects\CNN5\mixkit-office-telephone-ring-1350.wav")

face_clsfr=cv2.CascadeClassifier(r"C:\Users\Dell\Desktop\Shojeb ML Projects\CNN5\haarcascade_upperbody.xml")

app=Flask(__name__)
camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
# Load a sample picture and learn how to recognize it.
shojeb_image = face_recognition.load_image_file(r"C:\Users\Dell\Desktop\Shojeb ML Projects\CNN5\shojeb\shojeb.jpg")
shojeb_face_encoding = face_recognition.face_encodings(shojeb_image)[0]

# Load a second sample picture and learn how to recognize it.
shapnik_image = face_recognition.load_image_file(r"C:\Users\Dell\Desktop\Shojeb ML Projects\CNN5\shapnik\shapnik.jpg")
shapnik_face_encoding = face_recognition.face_encodings(shapnik_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    shojeb_face_encoding,
    shapnik_face_encoding
]
known_face_names = [
    "Army name:Shojeb",
    "Army name:Shapnik"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


labels_dict={0:'Army',1:'Local'}
color_dict={0:(0,255,0),1:(0,0,255)}

if not os.path.exists('data'):

    os.makedirs('data')  
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:

             # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

          
           
    # Find all the faces and face encodings in the current frame of video

    # Only process every other frame of video to save time
            process_this_frame = True
            if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

        process_this_frame = not process_this_frame


    
    
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  
        i=1
        for (x,y,w,h) in faces:
    
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]
        
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],4)
            cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],4)
            cv2.putText(frame, labels_dict[label]+str(i), (x, y-10),cv2.FONT_ITALIC, 1,(255,255,255),4)
        
            if(labels_dict[label] =='Army'):
                print("No Beep")
                cv2.putText(frame, name, (x, y+15),cv2.FONT_ITALIC, 1,(255,255,255),4)

                i += 1
                if (i<=2):
                    sound1.play()
                    print('alert! alert! No army here')
            # Display the results
            

            
            elif(labels_dict[label] =='Local'):
                sound.play()
                print("Beep")
                currentframe=0
 
                cv2.imwrite('./data/frame'+str(currentframe)+'.jpg',frame)
                currentframe+=1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
