import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# Initialize MTCNN with parameters you prefer
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    keep_all=True
)

# Capture video from the default camera (0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame (BGR in OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Get face detection results
    boxes, probs = mtcnn.detect(pil_img)
    
    # Draw bounding boxes around detected faces
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                # Draw rectangle on the original frame (BGR)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Show probability (confidence) above the box
                text = f"{prob:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow("Face Detection", frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()