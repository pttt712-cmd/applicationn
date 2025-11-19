import cv2
import math
import time
import threading
from collections import deque

# mediapipe
import mediapipe as mp

# TTS: try gTTS (better Korean) then fallback pyttsx3
GTTS_OK = False
try:
    from gtts import gTTS
    from playsound import playsound
    GTTS_OK = True
except Exception:
    GTTS_OK = False

import pyttsx3
import os
import tempfile

# ----------------------------
# TTS helper (non-blocking)
# ----------------------------
def _speak_sync(text):
    """Blocking speak function executed in a background thread."""
    if GTTS_OK:
        try:
            tmp = os.path.join(tempfile.gettempdir(), f"ksl_tts_tmp_{int(time.time()*1000)}.mp3")
            tts = gTTS(text=text, lang='ko')
            tts.save(tmp)
            try:
                # playsound blocks this thread but main thread is free
                playsound(tmp)
            except Exception as e:
                print("playsound error:", e)
            try:
                os.remove(tmp)
            except Exception:
                pass
            return
        except Exception as e:
            print("gTTS failed:", e, "→ fallback to pyttsx3")

    # fallback: create engine locally to avoid thread-safety issues with global engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        # Optional: try to set a Korean voice if available (uncomment & edit if you know voice id)
        # voices = engine.getProperty('voices')
        # for v in voices:
        #     print("VOICE:", v.id, v.name)
        # engine.setProperty('voice', 'com.apple.speech.synthesis.voice.yuna')  # example for macOS
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("pyttsx3 error:", e)

def speak_async(text):
    """Start a daemon thread to speak text (non-blocking)."""
    t = threading.Thread(target=_speak_sync, args=(text,), daemon=True)
    t.start()

# ----------------------------
# Angle utilities
# ----------------------------
def finger_angle(a, b, c):
    """Return angle ABC in degrees (points are landmarks with x,y,z)."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    mag1 = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    mag2 = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    if mag1*mag2 == 0:
        return 180.0
    cosv = dot / (mag1*mag2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

# ----------------------------
# Recognize sentence from single hand landmarks
# ----------------------------
def recognize_sentence(hand):
    """
    Input: hand is a mediapipe hand landmark object (one hand)
    Output: Korean sentence string or "" if none matched
    Method: compute finger angles and use simple pattern rules
    """
    lm = hand.landmark

    # define joints for angle calc: (MCP, PIP, TIP) approx by indices
    joints = {
        "thumb":  (lm[2], lm[3], lm[4]),
        "index":  (lm[5], lm[6], lm[8]),
        "middle": (lm[9], lm[10], lm[12]),
        "ring":   (lm[13], lm[14], lm[16]),
        "pinky":  (lm[17], lm[18], lm[20]),
    }

    angles = {}
    for f, (a,b,c) in joints.items():
        angles[f] = finger_angle(a,b,c)

    # finger considered "straight" if angle > 150 deg
    up = {f: (angles[f] > 150.0) for f in angles}

    # Additional: detect fist by checking tips are close to wrist in y direction
    wrist = lm[0]
    fist_score = 0
    for tip_idx in (8,12,16,20):
        if lm[tip_idx].y > lm[0].y:  # tip lower than wrist (camera coords)
            fist_score += 1

    # Patterns -> Korean phrases
    # 1) Hello: all five fingers up
    if all(up.values()):
        return "안녕하세요"

    # 2) Nice day / today is a beautiful day: index+middle up (like V) while ring/pinky down
    if up["index"] and up["middle"] and (not up["ring"]) and (not up["pinky"]):
        return "오늘은 참 좋은 날이에요"

    # 3) Like / Good: only thumb up
    if up["thumb"] and not up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
        return "좋아요"

    # 4) Thank you: (we map fist to thank you for convenience)
    if fist_score >= 4:
        return "감사합니다"

    # 5) I love you: index + pinky + thumb (rock sign with thumb)
    if up["index"] and up["pinky"] and up["thumb"] and (not up["middle"]) and (not up["ring"]):
        return "사랑해요"

    # 6) Sorry: fallback (note duplicate pattern with nice day: order matters)
    # Already covered by index+middle rule above, but kept for clarity
    # (If you need a distinct '미안해요' create a different geometric check)

    # 7) Hungry: hand pointing down (tips lower than wrist)
    tips_below = sum(1 for t in (8,12,16,20) if lm[t].y > lm[0].y)
    if tips_below >= 4 and (not any(up.values())):
        return "배고파요"

    # If nothing matched
    return ""

# ----------------------------
# Recognize from multi-hand (two-hand) patterns
# ----------------------------
def recognize_from_hands(multi_hand_landmarks):
    """
    Input: list of hand landmarks (0..2)
    Output: Korean sentence or ""
    Two-hand patterns:
     - both hands open -> greeting or 'beautiful day' depending on pose
     - other combos can be added
    """
    if not multi_hand_landmarks:
        return ""

    # If two hands present, use both
    if len(multi_hand_landmarks) >= 2:
        h1 = multi_hand_landmarks[0]
        h2 = multi_hand_landmarks[1]

        # compute straightness counts
        def count_open(hand):
            # count finger angles > 150
            cnt = 0
            try:
                for (a,b,c) in ((hand.landmark[2],hand.landmark[3],hand.landmark[4]),
                                (hand.landmark[5],hand.landmark[6],hand.landmark[8]),
                                (hand.landmark[9],hand.landmark[10],hand.landmark[12]),
                                (hand.landmark[13],hand.landmark[14],hand.landmark[16]),
                                (hand.landmark[17],hand.landmark[18],hand.landmark[20])):
                    ang = finger_angle(a,b,c)
                    if ang > 150:
                        cnt += 1
            except:
                pass
            return cnt

        open1 = count_open(h1)
        open2 = count_open(h2)

        # both hands open fully -> greeting
        if open1 >= 4 and open2 >= 4:
            return "안녕하세요"

        # both hands open wide (>=3) -> nice day
        if open1 >= 3 and open2 >= 3:
            return "오늘은 날씨가 참 좋네요"

        return ""

    # If only one hand present, fallback to single-hand recognize
    return recognize_sentence(multi_hand_landmarks[0])

# ----------------------------
# Main realtime loop
# ----------------------------
def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(max_num_hands=2,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened. Check permissions.")
        return

    # smoothing & debounce (reduced buffer for faster responsiveness)
    buf = deque(maxlen=6)  # was 10 -> now 6
    # keep last spoken text + timestamp to allow cooldown-based repeats
    last_spoken_text = ""
    last_spoken_ts = 0.0
    SPEAK_COOLDOWN = 1.5  # seconds before same phrase can be spoken again

    debug = False

    print("KSL Full App 시작: 'd' toggle debug. 'q' quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        detected = ""
        hands_present = False
        # multi-hand list
        if results.multi_hand_landmarks:
            hands_present = True
            # choose recognition based on number of hands
            if len(results.multi_hand_landmarks) >= 2:
                detected = recognize_from_hands(results.multi_hand_landmarks)
            else:
                detected = recognize_sentence(results.multi_hand_landmarks[0])

            # draw hands
            for h in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        else:
            # no hands detected -> we reset buffer for quicker next recognition
            # this allows immediate next gesture
            buf.clear()

        # smoothing
        buf.append(detected)
        # majority non-empty
        nonempty = [b for b in buf if b]
        final = ""
        stable = False
        if nonempty:
            # pick most common
            final = max(set(nonempty), key=nonempty.count)
            # require stability: appear at least half the buffer
            if nonempty.count(final) >= int(0.5 * len(buf)):
                stable = True

        # Show immediate detection for debugging (unstable in yellow, stable in green)
        if detected:
            color = (0,255,0) if stable else (0,200,255)
            cv2.putText(frame, detected, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        else:
            # optionally show last final for user, lightly
            if final:
                cv2.putText(frame, final, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180,180,180), 2)

        # speak when stable and cooldown elapsed and different enough
        now = time.time()
        if stable and final:
            can_speak = False
            if final != last_spoken_text:
                can_speak = True
            else:
                # same text; allow if cooldown elapsed
                if (now - last_spoken_ts) >= SPEAK_COOLDOWN:
                    can_speak = True

            if can_speak:
                print("SPEAK:", final)
                # non-blocking TTS
                speak_async(final)
                last_spoken_text = final
                last_spoken_ts = now

        # reset last_spoken_text quickly if no hands present so user can repeat gesture immediately
        if not hands_present:
            last_spoken_text = ""
            # small sleep not necessary; main loop continues

        # debug overlay
        if debug and results and results.multi_hand_landmarks:
            try:
                h0 = results.multi_hand_landmarks[0].landmark
                sample = f"idx_y:{h0[8].y:.3f} mid_y:{h0[12].y:.3f} ring_y:{h0[16].y:.3f}"
                cv2.putText(frame, sample, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,0), 2)
            except Exception:
                pass

        cv2.imshow("KSL Full App", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug = not debug
            print("Debug:", debug)

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()

if __name__ == "__main__":
    main()