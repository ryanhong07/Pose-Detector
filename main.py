import mediapipe as mp
import cv2
 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cam = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cam.isOpened():
        
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        # print(results.face_landmarks)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # face
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,230,0), thickness=1, circle_radius=1))
                                  
        # left hand
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,230,0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
        
        # right hand
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,230,0), thickness=2, circle_radius=2))
                
        # pose
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(15,219,213), thickness=2, circle_radius=2))
        
        cv2.imshow('Holistic Model Detections', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cam.release()
cv2.destroyAllWindows()

print('code completed')
 