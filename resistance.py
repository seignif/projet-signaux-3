import cv2
import numpy as np


def main():
    """
    Identifies color bands on a resistor from a live webcam feed and calculates its value.
    """
    # Define the standard resistor color code values (HSV lower and upper bounds)
    # Format: { 'color_name': ([H_lower, S_lower, V_lower], [H_upper, S_upper, V_upper], value) }
    # These HSV values may need tuning depending on your lighting conditions
    color_codes = {
        'black':  ([0, 0, 0], [180, 255, 40], 0),
        'brown':  ([8, 100, 20], [20, 255, 200], 1),
        'red':    ([0, 70, 50], [10, 255, 255], 2),
        'orange': ([10, 100, 100], [25, 255, 255], 3),
        'yellow': ([25, 100, 100], [35, 255, 255], 4),
        'green':  ([40, 40, 40], [90, 255, 255], 5),
        'blue':   ([90, 50, 50], [130, 255, 255], 6),
        'violet': ([130, 50, 50], [160, 255, 255], 7),
        'gray':   ([0, 0, 50], [180, 50, 220], 8),
        'white':  ([0, 0, 200], [180, 30, 255], 9),
    }

    # --- Initialize Webcam ---
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # --- Capture and Prepare Frame ---
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize for faster processing
        height, width, _ = frame.shape
        scale = 600 / width
        frame_resized = cv2.resize(frame, (600, int(height * scale)))
        output = frame_resized.copy()

        # Convert to HSV and apply Gaussian Blur
        hsv_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        blurred_img = cv2.GaussianBlur(hsv_img, (15, 15), 0)

        # --- Find the Resistor Body ---
        # NOTE: Adjust this range to match the color of your resistor's body
        light_brown_lower = np.array([10, 40, 90])
        light_brown_upper = np.array([30, 255, 255])
        mask = cv2.inRange(blurred_img, light_brown_lower, light_brown_upper)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            resistor_contour = max(contours, key=cv2.contourArea)
            # Only proceed if the contour is of a reasonable size
            if cv2.contourArea(resistor_contour) > 1000:
                x, y, w, h = cv2.boundingRect(resistor_contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                resistor_roi = hsv_img[y:y+h, x:x+w]

                # --- Identify Color Bands ---
                band_positions = []
                for color_name, (lower, upper, value) in color_codes.items():
                    lower_bound = np.array(lower)
                    upper_bound = np.array(upper)
                    
                    color_mask = cv2.inRange(resistor_roi, lower_bound, upper_bound)
                    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in color_contours:
                        # Filter out small noise contours and ensure bands are somewhat vertical
                        (bx, by, bw, bh) = cv2.boundingRect(c)
                        if bw < 30 and bh > 20:
                             M = cv2.moments(c)
                             if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                # Avoid detecting the same band multiple times
                                if not any(abs(cx - pos) < 10 for _, pos in band_positions):
                                    band_positions.append((color_name, cx))

                # --- Calculate and Display Value ---
                if len(band_positions) >= 3:
                    band_positions.sort(key=lambda item: item[1])
                    detected_bands = [item[0] for item in band_positions]

                    try:
                        val1 = color_codes[detected_bands[0]][2]
                        val2 = color_codes[detected_bands[1]][2]
                        multiplier_color = detected_bands[2]
                        multiplier = 10 ** color_codes[multiplier_color][2]
                        
                        resistance = (val1 * 10 + val2) * multiplier
                        
                        # Format the result string
                        if resistance >= 1_000_000:
                            result_str = f"{resistance / 1_000_000:.1f} MOhms"
                        elif resistance >= 1_000:
                            result_str = f"{resistance / 1_000:.1f} kOhms"
                        else:
                            result_str = f"{resistance} Ohms"

                        # Display the result
                        cv2.putText(output, result_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                    except (IndexError, KeyError):
                        pass # Fail silently if bands are misidentified in a frame

        # --- Display the final video feed ---
        cv2.imshow('Real-time Resistor Identifier', output)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Release resources ---
    cap.release()
    cv2.destroyAllWindows()


# --- Main Execution ---
if __name__ == "__main__":
    main()