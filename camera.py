import cv2
import time
from gpiozero import MotionSensor, LED
from picamera2 import Picamera2
from datetime import datetime
import pyrebase
import requests
import base64
import dlib


firebaseConfig = {
    'apiKey': "AIzaSyA0P0mvkBLIYW1N3BSeqfF0PMzm8WvjuYk",
    'authDomain': "smartcamera-e705c.firebaseapp.com",
    'databaseURL': "https://smartcamera-e705c-default-rtdb.firebaseio.com",
    'projectId': "smartcamera-e705c",
    'storageBucket': "smartcamera-e705c.appspot.com",
    'messagingSenderId': "692217413026",
    'appId': "1:692217413026:web:f1dab3b0572405d7e246fb",
    'measurementId': "G-RQVNB1Y805"
}

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()

pir = MotionSensor(4)
led = LED(17)


detector = dlib.get_frontal_face_detector()


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


url = 'http://127.0.0.1:8000/recog'
w, h = 100, 100 
response=''

camera = Picamera2()

while True:
    motion = {
        "status": "off"
    }

    db.child("PIR").update(motion)
        
    pir.wait_for_motion()
    print("Phat hien chuyen dong!")
    led.on()

    current_time = datetime.now().strftime("%d%m%Y%H%M%S")
    video_name = f"{current_time}.h264"
    camera.start_and_record_video(video_name, duration=10)
    print(f"Video {video_name} da luu.")
    led.off()
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print(f"Loi: khong the mo video {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_interval = int(fps * 2)  

    upload_to_firebase = True

    while cap.isOpened():
        for i in range(frame_interval - 1):  # Skip frames
            ret, _ = cap.read()
            if not ret:
                print("End of video.")
                cap.release()
                cv2.destroyAllWindows()
                break

        if not ret:
            break
            
        # Read the actual frame
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

         # Display the frame with rectangles
        cv2.imshow('Video', frame)

        # If faces are detected, encode the frame and send to the server
        if len(faces) > 0:
            encoded_image_data = encode_image_to_base64(frame)

            # Parameters to send in the POST request
            data = {
                'image': encoded_image_data,
                'w': w,
                'h': h
            }

            # Sending the POST request
            response = requests.post(url, data=data)
            #response = " Ngoc"

            # Printing the response
            print("Response from server:")
            print(response.text)
            
            clean_response = response.text.strip()
            if clean_response == 'Ngoc' or clean_response == 'Trinh' or clean_response == 'Nhi' : 
                upload_to_firebase = False

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if upload_to_firebase:

        try:
            storage.child(video_name).put(video_name)
            print("Video da tai len Firebase.")
        except Exception as e:
            print(f"Khong the tai video len Firebase: {e}")
        motion_data = {
            "status": "on"
        }
        
        try:
            db.child("PIR").update(motion_data)
            print("Luu tin hieu phat hien chuyen dong len firebase.")
        except Exception as e:
            print(f"Khong the luu tin hieu len fb: {e}")
            
           
    else:
            
        motion_data = {
            "status": "off"
        }
        try:
            db.child("PIR").update(motion_data)
            print("Khong phai nguoi la luu off len firebase.")
        except Exception as e:
            print(f"Khong the luu tin hieu len fb: {e}")
    

    
camera.close()
