import cv2
import pytesseract
from PIL import Image
import speed
# Initialize the video capture object
cap = cv2.VideoCapture('istock-804100844_preview.mp4')

# Initialize the number plate detector
detector = cv2.CascadeClassifier('haarcascade.xml')

# Initialize the number plate recognition system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Loop through each frame of the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break if end of video
    if not ret:
        break

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the number plates in the frame
    plates = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected number plate
    for (x, y, w, h) in plates:
        # Extract the ROI of the number plate
        roi = gray[y:y+h, x:x+w]
        plate_pil = Image.fromarray(roi)

        # Apply the OCR engine to the ROI
        text = pytesseract.image_to_string(plate_pil,  config='--psm 11')

        # Draw the bounding box and the extracted text on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Number Plate Detection', frame)

    # Break if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
