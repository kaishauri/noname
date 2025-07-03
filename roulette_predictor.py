#!/usr/bin/env python3 
import cv2
import numpy as np
import time
import csv
import math
import argparse
from collections import deque
from scipy import stats

EURO_NUMBERS_CLOCKWISE = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8,
    23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12,
    35, 3, 26
]
ANGLE_PER_POCKET = 360.0 / len(EURO_NUMBERS_CLOCKWISE)

def angle_from_center(pt, center):
    return (math.degrees(math.atan2(pt[1] - center[1], pt[0] - center[0])) + 360) % 360

def detect_wheel_center(gray_frame):
    circles = cv2.HoughCircles(
        gray_frame, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=gray_frame.shape[0] // 2,
        param1=50, param2=30,
        minRadius=gray_frame.shape[0] // 6,
        maxRadius=gray_frame.shape[0] // 2
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, _ = circles[0][0]
        return int(cx), int(cy)
    return None

def estimate_wheel_angle(gray_roi, ref_pattern):
    shift, _ = cv2.phaseCorrelate(np.float32(ref_pattern), np.float32(gray_roi))
    dx = shift[0]
    angle = (dx / gray_roi.shape[1]) * 360.0
    return angle % 360

def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.7
    return cv2.SimpleBlobDetector_create(params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=60, help="FPS for processing and timing")
    ap.add_argument("--outfile", help="MP4 filename to save output video (optional)")
    ap.add_argument("--video", help="Path to input video file (required)")
    args = ap.parse_args()

    if not args.video:
        print("[ERROR] Please provide a path to an input video file using --video")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = args.fps
    frame_interval = 1.0 / fps

    ret, ref_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame from video")
        return

    h, w = ref_frame.shape[:2]
    rim_thickness = int(h * 0.15)

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_rim = ref_gray[-rim_thickness:, :]

    blob_detector = create_blob_detector()

    vw = None
    if args.outfile:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.outfile, fourcc, fps, (w, h))
        if not vw.isOpened():
            print(f"[ERROR] Could not open video writer for {args.outfile}")
            vw = None
        else:
            print(f"[INFO] Recording video to {args.outfile}")

    csv_file = open("predictions_log.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Ball Angle", "Wheel Angle", "Predicted Number"])

    ball_angles = deque(maxlen=120)
    wheel_angles = deque(maxlen=120)
    timestamps = deque(maxlen=120)

    last_pred = None
    print("[INFO] Starting video processing. Press ESC to quit.")

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            keypoints = blob_detector.detect(blur)
            ang_ball = None
            if keypoints:
                kp = keypoints[0]
                cx, cy = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame, (cx, cy), int(kp.size / 2), (0, 255, 0), 2)
                center = detect_wheel_center(gray) or (w // 2, h // 2)
                ang_ball = angle_from_center((cx, cy), center)

            gray_rim = gray[-rim_thickness:, :]
            ang_wheel = estimate_wheel_angle(gray_rim, ref_rim)

            now = time.time()
            ball_angles.append(ang_ball if ang_ball is not None else (ball_angles[-1] if ball_angles else 0))
            wheel_angles.append(ang_wheel)
            timestamps.append(now)

            if ang_ball is not None:
                cv2.putText(frame, f"Ball Angle: {ang_ball:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Wheel Angle: {ang_wheel:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            if len(ball_angles) > fps:
                t = np.array(timestamps) - timestamps[0]
                ball = np.unwrap(np.radians(ball_angles))
                wheel = np.unwrap(np.radians(wheel_angles))
                Nfit = min(30, len(t))
                slope_ball, _, _, _, _ = stats.linregress(t[-Nfit:], ball[-Nfit:])
                slope_wheel, _, _, _, _ = stats.linregress(t[-Nfit:], wheel[-Nfit:])

                decel = -1.0
                if len(t) > Nfit + 10:
                    dv = slope_ball - (ball[-Nfit] - ball[-Nfit-10]) / (t[-Nfit] - t[-Nfit-10])
                    decel = dv / (t[-1] - t[-Nfit-10])

                if decel < 0:
                    t_hit = (abs(slope_ball) - abs(slope_wheel)) / abs(decel)
                    if 0 < t_hit < 10:
                        ball_future = ball[-1] + slope_ball * t_hit + 0.5 * decel * t_hit**2
                        wheel_future = wheel[-1] + slope_wheel * t_hit
                        rel_angle = (math.degrees(ball_future - wheel_future) % 360)
                        pocket_index = int(rel_angle // ANGLE_PER_POCKET)
                        last_pred = EURO_NUMBERS_CLOCKWISE[pocket_index]

            if last_pred is not None:
                cv2.putText(frame, f"PREDICTED: {last_pred}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if ang_ball is not None:
                    csv_writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{ang_ball:.2f}",
                        f"{ang_wheel:.2f}",
                        last_pred
                    ])

            cv2.imshow("Roulette Tracker", frame)

            if vw is not None:
                vw.write(frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        if vw is not None:
            vw.release()
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()