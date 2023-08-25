import cv2
from deepface import DeepFace

# load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")
# initialize the camera
cap = cv2.VideoCapture(0)

# loop over frames from the camera
while True:
    # read a frame from the camera
    ret, frame = cap.read()
    
    # detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame, 1.3, 5)
    
    # loop over the faces
    for (x,y,w,h) in faces:
        # extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # preprocess the face ROI for the emotion detection model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi.astype("float") / 255.0
        face_roi = face_roi.reshape((1, 48, 48, 1))
        
        # predict the emotion using the model
        preds = model.predict(face_roi)[0]
        emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][preds.argmax()]
        
        # draw the emotion label on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow("Emotion Detection", frame)

    
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()