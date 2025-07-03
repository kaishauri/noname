import cv2
import numpy as np
import mss
import time
from utils.tracking import find_ball_position
from utils.speed import calculate_speed

positions = []
timestamps = []

def main():
    print("Starting roulette tracker...")
    cv2.namedWindow("Roulette Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Roulette Tracker", 800, 600)
    
    with mss.mss() as sct:
        # Define screen capture area (adjust to your game)
        monitor = {"top": 100, "left": 100, "width": 640, "height": 480}
        
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Detect ball position
            ball_pos, confidence = find_ball_position(frame)

            if ball_pos:
                positions.append(ball_pos)
                timestamps.append(time.time())

                if len(positions) >= 2:
                    speed = calculate_speed(positions[-2], positions[-1], timestamps[-2], timestamps[-1])
                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.0f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw result
            if ball_pos:
                # Draw ball position with confidence indicator
                cv2.circle(frame, ball_pos, 5, (0, 0, 255), -1)
                cv2.circle(frame, ball_pos, 10, (0, 0, 255), 2)
                
                # Draw confidence indicator
                if confidence > 500:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 200:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                cv2.circle(frame, ball_pos, 15, color, 2)

            # Show the frame
            cv2.imshow("Roulette Tracker", frame)
            
            # Break the loop if 'q' or 'esc' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    # Cleanup
    cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
