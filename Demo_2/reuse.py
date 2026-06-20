import cv2
import numpy as np
from ultralytics import YOLO
import Jetson.GPIO as GPIO
import time

# --- Hardware & PWM Setup ---
BUZZER_PIN = 33
LED_PIN = 31  

GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# Initialize PWM: (Pin, Frequency in Hz)
pwm = GPIO.PWM(BUZZER_PIN, 880)

# --- Pulsing Alarm Variables ---
ALARM_INTERVAL = 0.25       # How fast the alarm pulses (in seconds). 0.25 = 4 pulses a second
last_alarm_time = 0         # Tracks the last time the LED/Buzzer toggled
alarm_hardware_on = False   # Tracks if the LED/Buzzer are currently physically ON
alarm_mode_active = False   # Tracks if we are in the "Alarm State" overall

# --- Buzzer Logic Config ---
REQUIRED_FRAMES = 100   
GRACE_THRESHOLD = 10   
detection_counter = 0
missed_frames = 0

# --- YOLO Optimization Config ---
model = YOLO('./best.engine', task='detect')
#model.to('cuda')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

PIXEL_CHANGE_THRESH = 25
REGION_CHANGE_RATIO = 0.10
FULL_REFRESH_INTERVAL = 20

prev_gray = None
cached_detections = []
cached_regions = []
frame_id = 0

# --- Helper Functions ---
def run_yolo(frame):
    results = model(frame, conf=0.85, device=0, verbose=False)
    return [tuple(map(int, box.xyxy[0])) for box in results[0].boxes]

def extract_region(gray, x1, y1, x2, y2, h, w):
    return gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

def region_changed(prev_crop, curr_gray, x1, y1, x2, y2, h, w):
    curr_crop = curr_gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if prev_crop.shape != curr_crop.shape or prev_crop.size == 0: return True
    diff = cv2.absdiff(prev_crop, curr_crop)
    return (np.count_nonzero(diff > PIXEL_CHANGE_THRESH) / max(diff.size, 1)) > REGION_CHANGE_RATIO

def compute_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter
    return inter / max(union, 1)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # --- Computation Reuse Logic ---
        if prev_gray is None or (frame_id % FULL_REFRESH_INTERVAL == 0):
            cached_detections = run_yolo(frame)
            reused_flags = [False] * len(cached_detections)
        else:
            still_valid, stale = [], []
            for i, (det, crop) in enumerate(zip(cached_detections, cached_regions)):
                if region_changed(crop, curr_gray, *det, h, w): stale.append(i)
                else: still_valid.append(i)

            if not stale and cached_detections:
                reused_flags = [True] * len(cached_detections)
            else:
                new_dets = run_yolo(frame)
                reused_boxes = [cached_detections[i] for i in still_valid]
                final_dets = list(reused_boxes)
                final_flags = [True] * len(reused_boxes)
                for nb in new_dets:
                    if not any(compute_iou(nb, rb) > 0.5 for rb in reused_boxes):
                        final_dets.append(nb); final_flags.append(False)
                cached_detections, reused_flags = final_dets, final_flags

        cached_regions = [extract_region(curr_gray, *d, h, w) for d in cached_detections]

        # --- Detection Counter Logic ---
        if cached_detections:
            detection_counter += 1
            missed_frames = 0
        else:
            missed_frames += 1
            if missed_frames > GRACE_THRESHOLD:
                detection_counter = 0

        # --- Pulsing Alarm Logic ---
        if detection_counter >= REQUIRED_FRAMES:
            alarm_mode_active = True
            current_time = time.time()
            
            # Check if enough time has passed to toggle the alarm state
            if current_time - last_alarm_time >= ALARM_INTERVAL:
                alarm_hardware_on = not alarm_hardware_on # Flip the state
                last_alarm_time = current_time            # Reset the timer
                
                # Apply the toggled state to the hardware
                if alarm_hardware_on:
                    pwm.start(50)
                    GPIO.output(LED_PIN, GPIO.HIGH)
                else:
                    pwm.stop()
                    GPIO.output(LED_PIN, GPIO.LOW)
                    
            status_color = (0, 0, 255) # Red text for alert
            
        else:
            # Only send the STOP commands if we were just in an alarm state
            if alarm_mode_active:
                pwm.stop()
                GPIO.output(LED_PIN, GPIO.LOW)
                
                # Reset alarm variables
                alarm_mode_active = False
                alarm_hardware_on = False 
                
            status_color = (0, 255, 0) # Green text for clear

        # --- Visuals ---
        cv2.putText(display, f"Detection Streak: {detection_counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        for det, reused in zip(cached_detections, reused_flags):
            color = (0, 165, 255) if reused else (0, 255, 0)
            cv2.rectangle(display, (det[0], det[1]), (det[2], det[3]), color, 2)

        cv2.imshow("Passive Buzzer Reuse System", display)
        prev_gray, frame_id = curr_gray, frame_id + 1
        if cv2.waitKey(1) == 27: break

finally:
    # --- Safe Cleanup ---
    print("Cleaning up resources...")
    pwm.stop()
    GPIO.output(LED_PIN, GPIO.LOW) 
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()