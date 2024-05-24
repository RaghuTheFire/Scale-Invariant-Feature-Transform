
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

int main() 
{
    // Initialize video capture
    cv::VideoCapture cap("plane.mp4");

    // Initialize SIFT detector and matcher
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create();

    // Initialize variables
    cv::Rect roi;
    bool roi_selected = false;
    bool tracking = false;
    cv::Mat prev_frame;
    std::vector<cv::KeyPoint> prev_kp;
    cv::Mat prev_desc;

    // Mouse callback function
    cv::Rect roi_rect;
    void select_roi(int event, int x, int y, int flags, void* param) 
  {
        if (event == cv::EVENT_LBUTTONDOWN) 
        {
            roi_rect.x = x;
            roi_rect.y = y;
        } 
        else 
          if (event == cv::EVENT_LBUTTONUP) 
          {
            roi_rect.width = x - roi_rect.x;
            roi_rect.height = y - roi_rect.y;
            roi_selected = true;
        }
    }

    // Main loop
    while (true) 
    {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret) 
        {
            break;
        }

        // Draw ROI selection rectangle
        if (roi_selected) 
        {
            cv::rectangle(frame, roi_rect, cv::Scalar(0, 255, 0), 2);
        }

        // Display the frame
        cv::imshow("Frame", frame);

        // Handle user input
        int key = cv::waitKey(1) & 0xFF;
        if (key == 's') 
        {
            roi_selected = false;
            tracking = true;
            roi = roi_rect;
            prev_frame = frame(roi);
            std::vector<cv::KeyPoint> kp;
            cv::Mat desc;
            sift->detectAndCompute(prev_frame, cv::noArray(), kp, desc);
            prev_kp = kp;
            prev_desc = desc;
        } 
        else 
        if (key == 'q') 
        {
            break;
        }

        // Track the ROI
        if (tracking) 
        {
            cv::Mat curr_frame = frame(roi);
            std::vector<cv::KeyPoint> curr_kp;
            cv::Mat curr_desc;
            sift->detectAndCompute(curr_frame, cv::noArray(), curr_kp, curr_desc);
            std::vector<std::vector<cv::DMatch>> matches;
            bf->knnMatch(prev_desc, curr_desc, matches, 2);
            std::vector<cv::DMatch> good_matches;
            for (const auto& match : matches) 
            {
                if (match[0].distance < 0.7 * match[1].distance) 
                {
                    good_matches.push_back(match[0]);
                }
            }

            if (good_matches.size() > 10) 
            {
                std::vector<cv::Point2f> src_pts, dst_pts;
                for (const auto& match : good_matches) 
                {
                    src_pts.push_back(prev_kp[match.queryIdx].pt);
                    dst_pts.push_back(curr_kp[match.trainIdx].pt);
                }
                cv::Mat M = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0);
                std::vector<cv::Point2f> pts = {cv::Point2f(0, 0), cv::Point2f(0, prev_frame.rows), cv::Point2f(prev_frame.cols, prev_frame.rows), cv::Point2f(prev_frame.cols, 0)};
                std::vector<cv::Point2f> dst;
                cv::perspectiveTransform(pts, M, dst);
                cv::polylines(frame, dst, true, cv::Scalar(255, 0, 0), 2);
                prev_frame = curr_frame.clone();
                prev_kp = curr_kp;
                prev_desc = curr_desc;
            }
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

