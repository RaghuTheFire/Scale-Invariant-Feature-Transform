# ObjectTracking-SIFT-Algorithm

Scale-Invariant Feature Transform (SIFT) is a powerful computer vision algorithm designed to detect, describe, and match local features in images and videos.  It operates by identifying distinctive key points (interest points) that are invariant to changes in scale, rotation, and illumination. These key points serve as robust descriptors for recognizing objects and patterns.

# Reference
https://medium.com/@siromermer/object-tracking-with-sift-algorithm-using-opencv-51be3c6882c9

This C++ and Python code and uses the OpenCV library for image processing and feature detection. It performs the following steps:

1. Opens a video file and reads the first frame.
2. Allows the user to select a region of interest (ROI) by drawing a rectangle using mouse events.
3. Extracts the ROI from the first frame and converts it to grayscale.
4. Opens the video again and creates a SIFT object and a BFMatcher object.
5. Finds the keypoints and descriptors for the ROI using SIFT.
6. Processes each frame of the video:
   - Converts the frame to grayscale.
   - Finds the keypoints and descriptors for the frame using SIFT.
   - Matches the descriptors of the ROI with the descriptors of the frame using the BFMatcher.
   - Draws circles on the frame at the locations of the matched keypoints.
   - Displays the frame with the drawn circles.
   - Allows the user to exit the loop by pressing the Esc key.

