import cv2
import mediapipe as mp
import math
import time

# 1. Initialize MediaPipe Hands and Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Configuration for Word/Sentence Building ---
# The word buffer stores the recognized letters
WORD_BUFFER = []
# Minimum number of consecutive frames a gesture must be stable to be added
STABILITY_THRESHOLD = 20
# Time delay (in seconds) after a gesture is recorded before accepting a new one
COOLDOWN_TIME = 0.5 
# Stores the last time a character was recorded
LAST_RECORD_TIME = time.time()
# Stores the currently detected, stable gesture
STABLE_GESTURE = ""
# Counts how many frames the current gesture has been stable
STABILITY_COUNT = 0


# Landmark Indices
TIP_IDS = [4, 8, 12, 16, 20] 

# --- Utility Functions ---

def get_landmark_coords(hand_landmarks, width, height):
    """Converts normalized landmarks to pixel coordinates."""
    lm_list = []
    for lm in hand_landmarks.landmark:
        # Convert normalized coordinates (0 to 1) to pixel coordinates
        lm_list.append((int(lm.x * width), int(lm.y * height)))
    return lm_list

def is_finger_extended(lm_list, tip_id, pip_id):
    """Checks if a non-thumb finger is extended by comparing Y-coordinates."""
    # Tip (e.g., 8) is 'up' if its Y-coordinate is smaller (higher on screen) than its MCP (e.g., 5)
    return lm_list[tip_id][1] < lm_list[pip_id - 1][1] # Compare tip to MCP for robust check

def is_thumb_extended(lm_list, handedness):
    """Checks if the thumb is extended by comparing X-coordinates."""
    is_right_hand = handedness.classification[0].label == 'Right'
    # Thumb Tip (4) is compared to the MCP joint (2)
    
    if is_right_hand:
        # Right hand: Tip (4) is further left (smaller X) if extended.
        return lm_list[4][0] < lm_list[2][0]
    else:
        # Left hand: Tip (4) is further right (larger X) if extended.
        return lm_list[4][0] > lm_list[2][0]

# --- ASL Classification (Enhanced for Letter Recognition) ---

def classify_asl_sign(lm_list, handedness):
    """Classifies a simplified ASL sign based on joint positions."""
    if not lm_list:
        return "Unknown"

    is_right_hand = handedness.classification[0].label == 'Right'
    
    # Check extension status for all 5 fingers
    fingers_up = [
        is_thumb_extended(lm_list, handedness),
        is_finger_extended(lm_list, 8, 6),   # Index Tip (8) to PIP (6)
        is_finger_extended(lm_list, 12, 10), # Middle Tip (12) to PIP (10)
        is_finger_extended(lm_list, 16, 14), # Ring Tip (16) to PIP (14)
        is_finger_extended(lm_list, 20, 18)  # Pinky Tip (20) to PIP (18)
    ]
    
    thumb_up, index_up, middle_up, ring_up, pinky_up = fingers_up
    
    # Simple check for a fully closed fist (most tips below their knuckles)
    closed_count = sum(1 for up in fingers_up if not up)
    
    # --- Classification Rules (Targeting letters from your chart) ---
    
    # Sign 'A' (Fist, thumb at the side/front)
    if closed_count == 5:
        # Check if thumb tip is roughly below the index MCP (5)
        if lm_list[4][1] > lm_list[5][1] and lm_list[4][0] < lm_list[9][0]:
            return "A"
        
    # Sign 'S' (Fist, thumb across index/middle)
    if closed_count == 5:
        # Check if thumb tip is significantly higher (smaller Y) than index MCP (5)
        if lm_list[4][1] < lm_list[5][1]:
            return "S" # A vs S is tricky, this is a simple differentiator

    # Sign 'B' (Four fingers straight up, thumb across palm)
    if index_up and middle_up and ring_up and pinky_up and not thumb_up:
        return "B"

    # Sign 'C' (Curved hand) - complex, using a simple height ratio
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        # Check if tips are slightly higher than their mid-knuckles (PIP) but lower than MCP (not a full B)
        # This is a highly simplified rule.
        if lm_list[8][1] < lm_list[9][1] and lm_list[12][1] < lm_list[13][1]:
            return "C"

    # Sign 'D' (Index finger up, others closed, thumb tucked)
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "D"
            
    # Sign 'F' (Three fingers up, index/thumb form a circle)
    if middle_up and ring_up and pinky_up and not index_up:
        # Check if thumb is touching or near the index finger tip (4 close to 8)
        if math.dist(lm_list[4], lm_list[8]) < 60: # Threshold adjusted for pixel distance
             return "F"
             
    # Sign 'G' (Index finger pointing, hand sideways) - requires 3D coordinates (Z) for accuracy.
    # We will use the V shape and check for verticality as a fallback for 2D.
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
         return "G / H" # 'G' and 'H' are easily confused in 2D
        
    # Sign 'I' (Pinky finger extended)
    if pinky_up and not index_up and not middle_up and not ring_up and not thumb_up:
        return "I"

    # Sign 'L' (Index and Thumb up)
    if index_up and thumb_up and not middle_up and not ring_up and not pinky_up:
        return "L"

    # Sign 'V' (Index and Middle fingers extended)
    if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        return "V"
            
    # Sign 'W' (Index, Middle, Ring up) - Matches your original '3' image
    if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
        return "W"
        
    # Sign 'Y' (Pinky and Thumb extended)
    if pinky_up and thumb_up and not index_up and not middle_up and not ring_up:
        return "Y"
        
    # Sign 'Space' (Open hand, used for spacing in words)
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "Space"
        
    # Fallback/Unclassified sign
    return "Unknown"


# 2. Main Loop for Real-Time Video Capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    
    while cap.isOpened():
        current_time = time.time()
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1) # Flip for selfie-view
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        current_gesture = "Unknown"
        
        # 3. Process Detected Hands (We only use the first hand for word building)
        if results.multi_hand_landmarks:
            
            # Use the first detected hand for classification
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0]
            
            # Get coordinates
            lm_list = get_landmark_coords(hand_landmarks, width, height)
            
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

            # Classify the ASL gesture
            current_gesture = classify_asl_sign(lm_list, handedness)
            
            # Get the wrist landmark for text positioning
            wrist_x, wrist_y = lm_list[0]
            
            # Display the current gesture on the video feed
            cv2.putText(
                image,
                f"Sign: {current_gesture}",
                (wrist_x - 50, wrist_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

        # 4. Word/Sentence Accumulation Logic
        
        # Check for stability and cooldown
        if current_gesture != "Unknown":
            if current_gesture == STABLE_GESTURE:
                STABILITY_COUNT += 1
            else:
                STABLE_GESTURE = current_gesture
                STABILITY_COUNT = 0
            
            # If the gesture is stable for enough frames AND cooldown is over
            if STABILITY_COUNT >= STABILITY_THRESHOLD and (current_time - LAST_RECORD_TIME) > COOLDOWN_TIME:
                
                # Special handling for "Space"
                if current_gesture == "Space":
                    if WORD_BUFFER and WORD_BUFFER[-1] != " ":
                        WORD_BUFFER.append(" ")
                # Handle letters
                elif current_gesture not in ["Unknown", "Space"]:
                    WORD_BUFFER.append(current_gesture)

                LAST_RECORD_TIME = current_time
                STABILITY_COUNT = 0 # Reset count after recording
        else:
            # Reset stability if no hand is detected or it's unknown
            STABLE_GESTURE = ""
            STABILITY_COUNT = 0

        # 5. Drawing the Word/Sentence Box (Like in your uploaded image)
        
        # Define the position for the word display box
        WORD_BOX_X = 50
        WORD_BOX_Y = 50
        
        # Join the buffer to form the current word/sentence
        current_word = "".join(WORD_BUFFER)
        
        # Draw a translucent box (optional, but looks professional)
        overlay = image.copy()
        cv2.rectangle(overlay, (WORD_BOX_X - 10, WORD_BOX_Y - 40), 
                      (WORD_BOX_X + len(current_word) * 25 + 20, WORD_BOX_Y + 20), 
                      (50, 50, 50), -1)
        alpha = 0.6
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Display the accumulated word/sentence
        cv2.putText(
            image,
            current_word,
            (WORD_BOX_X, WORD_BOX_Y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )

        # 6. Display the instructions/status
        status_text = f"Hold sign for {STABILITY_THRESHOLD} frames ({STABILITY_COUNT}) to record. Press 'c' to Clear."
        cv2.putText(image, status_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        # 7. Display the output frame
        cv2.imshow('ASL Sentence Builder', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            WORD_BUFFER = [] # Clear the word buffer

# 8. Cleanup
cap.release()
cv2.destroyAllWindows()