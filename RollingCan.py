"""This program processes a video to detect cans or circular objects using Hough Circle Transform.
It reads frames from a video, rotates them, resizes them for improved circle detection accuracy, and applies grayscale conversion and Gaussian blurring.
It then detects circles in the frames using the Hough Circle Transform method, scales the detected circles back to the original size, 
and draws the circles and their centers on the frames. The program continues processing the video until the user presses 'q' to quit."""



# Resizes the camera frame while maintaining aspect ratio
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):  
    dim = None  # Initialize the dimensions to None  
    (h, w) = image.shape[:2]

    if width is None and height is None:  
        return image, 1.0

    if width is None:  
        r = height / float(h)
        dim = (int(w * r), height) 
    else:  
        r = width / float(w)
        dim = (width, int(h * r)) 

    resized_image = cv2.resize(image, dim, interpolation=inter)  # Resize the image using the calculated dimensions  
    return resized_image, r  # Return the resized image and scaling ratio


# Function to detect the can in the video
def detect_cans_in_video():  
    cap = cv2.VideoCapture("rollcan.mp4")  # Open the video file for processing  

    while True:  
        ret, frame = cap.read()  # Read each frame from the video  
        if not ret:  # If the frame cannot be read, break the loop  
            break  
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate the frame 90 degrees for proper orientation  
        
        # Resize frame to improve circle detection accuracy
        resized_frame, scale_factor = resize_with_aspect_ratio(frame, width=300)  # Resize and get scaling factor  

        # Convert frame to grayscale and apply Gaussian blur for noise reduction  
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)  
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)  # Apply Gaussian blur  

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(  
            blurred,  
            cv2.HOUGH_GRADIENT,  
            dp=1.2,  
            minDist=25,  
            param1=120,  
            param2=100,  
            minRadius=0,  
            maxRadius=0  
        )  

        overlay_frame = frame.copy()  # Create a copy of the original frame to overlay detected circles

        if circles is not None:  # If circles are detected, process them  
            circles = np.uint16(np.around(circles[0, :]))  # Convert detected circles to integers

            for circle in circles:  # Loop through each detected circle  
                x, y, r = circle  # Get the x, y coordinates and radius of the circle  
                # Scale the circle coordinates and radius back to the original size  
                x = int(x * (1 / scale_factor))  
                y = int(y * (1 / scale_factor))  
                r = int(r * (1 / scale_factor))  

                # Draw the circle and its center on the frame
                cv2.circle(overlay_frame, (x, y), r, (0, 255, 0), 3)  # Draw the outer circle  
                cv2.circle(overlay_frame, (x, y), 2, (0, 0, 255), 3)  # Draw the center of the circle  

        # Show the video frame with the circles drawn on it
        cv2.imshow("Original Video with Circle Detection", overlay_frame)  

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  # Release the video capture object  
    cv2.destroyAllWindows()  # Close all OpenCV windows


# Start the program
detect_cans_in_video()  # Call the function to start detecting cans in the video
