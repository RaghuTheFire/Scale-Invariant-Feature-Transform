# ObjectTracking-SIFT-Algorithm

Scale-Invariant Feature Transform (SIFT) is a powerful computer vision algorithm designed to detect, describe, and match local features in images and videos.  It operates by identifying distinctive key points (interest points) that are invariant to changes in scale, rotation, and illumination. These key points serve as robust descriptors for recognizing objects and patterns.

# Reference
https://medium.com/@siromermer/object-tracking-with-sift-algorithm-using-opencv-51be3c6882c9

# SIFTTracker.cpp
This C++ uses the OpenCV library for image processing and feature detection. It performs the following steps:
 Here's a breakdown of the code: 
 1. The necessary OpenCV libraries are included: `opencv2/opencv.hpp` and `opencv2/features2d.hpp`.
2. The `main` function is the entry point of the program.
3. A `VideoCapture` object is created to read the video file specified by the path `"path/to/video.mp4"`.
4. The SIFT feature detector and the Brute-Force Matcher (BFMatcher) are initialized.
5. Several variables are declared to store the ROI, previous frame, keypoints, and descriptors.
6. A mouse callback function `select_roi` is defined to allow the user to select the ROI by clicking and dragging the mouse.
7. The main loop begins, where each frame of the video is read and processed.
8. If the ROI is selected, it is drawn on the frame using a green rectangle.
9. The frame is displayed in a window named "Frame".
10. The user can press 's' to start tracking the selected ROI or 'q' to quit the program.
11. If tracking is enabled, the following steps are performed:
a. The current frame's ROI is extracted.
b. SIFT keypoints and descriptors are computed for the current ROI.
c. The k-Nearest Neighbors (kNN) algorithm is used to match the descriptors between the previous and current frames.
d. Good matches are filtered based on a ratio test.
e. If there are enough good matches, a homography matrix is computed using RANSAC.
f. The homography matrix is used to transform the ROI from the previous frame to the current frame.
g. The transformed ROI is drawn on the current frame using a red polygon.
h. The previous frame, keypoints, and descriptors are updated for the next iteration.

12. After the main loop ends, the video capture object is released, and all windows are closed. In summary, this code allows the user to select a region of interest in a video, and then it tracks that region across subsequent frames using SIFT feature detection and matching, along with homography estimation to account for perspective changes.
