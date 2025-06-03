import time
import cv2
from deepface import DeepFace

def initialize_camera():
    return cv2.VideoCapture(0)

def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#we can substitute this with any library that we choose
def analyze_emotion(face_image):
    try:
        analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        return analysis[0].get('dominant_emotion', 'unknown')
    except Exception:
        return 'error'

def draw_results(frame, face_coordinates, emotion):
    x, y, w, h = face_coordinates
    cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

def main():
    camera = initialize_camera()
    face_detector = load_face_detector()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
        faces = face_detector.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Select the biggest face based on area (w * h)
            biggest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = biggest_face
            face_img = rgb_frame[y:y+h, x:x+w]
            emotion = analyze_emotion(face_img)
            draw_results(frame, (x, y, w, h), emotion)

        cv2.imshow("Emotion Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #functionality to simulate the requests coming in at a slower rate
        #time.sleep(0.1)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
