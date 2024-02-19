import cv2
import numpy as np

# define the video file path
video_file = "Freewa.mp4"

# Define the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()
scale_factor = 0.5
# Define the video capture object
cap = cv2.VideoCapture(video_file)

# Define the variable to keep track of the previous frame
previous_frame = None

# Define the scale factor (pixels per meter)
scale_factor = 0.05  # Change this value based on your video properties

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to the frame
    fgmask = fgbg.apply(gray)

    # Apply morphology operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Compute the contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and check if they are vehicles
    for contour in contours:
        # Compute the contour area
        area = cv2.contourArea(contour)

        # Check if the area is greater than a threshold (to filter out noise)
        if area > 100:
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Check if there is a previous frame to compare with
            if previous_frame is not None:
                # Compute the optical flow between the previous and current frames
                flow = cv2.calcOpticalFlowFarneback(previous_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Compute the speed of the vehicle in pixels per frame
                speed_pixels_per_frame = np.mean(flow[y:y+h, x:x+w, 0])

                # Convert the speed from pixels per frame to kilometers per hour
                speed_kph = speed_pixels_per_frame * scale_factor * 30 * 3.6  # Change 30 to a more appropriate value based on your video properties

                # Display the speed of the vehicle on the frame
                cv2.putText(frame, "{:.2f} km/h".format(speed_kph), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with the bounding boxes and speeds
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Store the current frame as the previous frame
    previous_frame = gray

# Release the video capture object
cap.release()


cv2.destroyAllWindows()