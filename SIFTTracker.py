
import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('plane.mp4')

# Initialize SIFT detector and matcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Initialize variables
roi = None
roi_selected = False
tracking = False
prev_frame = None
prev_kp = None
prev_desc = None

# Mouse callback function
def select_roi(event, x, y, flags, param):
    global roi, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = [x, y, 0, 0]
    elif event == cv2.EVENT_LBUTTONUP:
        roi[2] = x - roi[0]
        roi[3] = y - roi[1]
        roi_selected = True

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI selection rectangle
    if roi_selected:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Handle user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        roi_selected = False
        tracking = True
        roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        kp, desc = sift.detectAndCompute(roi_frame, None)
        prev_frame = roi_frame
        prev_kp = kp
        prev_desc = desc
    elif key == ord('q'):
        break

    # Track the ROI
    if tracking:
        curr_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        curr_kp, curr_desc = sift.detectAndCompute(curr_frame, None)
        matches = bf.knnMatch(prev_desc, curr_desc, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = prev_frame.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 2)
            prev_frame = curr_frame.copy()
            prev_kp = curr_kp
            prev_desc = curr_desc

# Release resources
cap.release()
cv2.destroyAllWindows()
