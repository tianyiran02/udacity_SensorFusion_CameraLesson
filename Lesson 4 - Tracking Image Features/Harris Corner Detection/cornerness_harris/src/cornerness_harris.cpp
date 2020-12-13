#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

static bool NMS(int column, int row, const cv::Mat & img, int limit)
{
    int temp, tempMax, tempCmp;
    int searchColLow, searchRowLow, searchColHigh, searchRowHigh;

    /* Determine the maximum search row and column */
    temp = (limit - 1) / 2;

    column >= temp ? searchColLow = column - temp : searchColLow = 0;
    row >= temp ? searchRowLow = row - temp : searchRowLow = 0;

    tempMax = column + temp;
    tempMax > img.cols ? searchColHigh = (img.cols - 1) : searchColHigh = tempMax;
    tempMax = row + temp;
    tempMax > img.rows ? searchRowHigh = (img.rows - 1) : searchRowHigh = tempMax;

    /* Now determine whether current point is local maximum */
    temp = img.at<unsigned char>(column, row);
    for (int i = searchColLow; i <= searchColHigh; i++)
    {
        for (int j = searchRowLow; j < searchRowHigh; j++)
        {
            tempCmp = img.at<unsigned char>(j, i);
            if (tempCmp > temp)
            {
                /* Directly return if another point is larger */
                return false;
            }
        }
    }

    std::cout << "Key point found!\n    column: " << column << "\n    row: " << row << std::endl;
    return true;
}

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    std::cout << "Determine the type of data: " << dst_norm_scaled.type() << std::endl;

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    int size = 3;   /* size must be odd, and greater than 1 (minimum 3) */
    bool hit = false;
    std::vector<cv::KeyPoint> logKeypoints;

    std::cout << "total rows: " << dst_norm_scaled.rows << std::endl;
    std::cout << "total columns: " << dst_norm_scaled.cols << std::endl;

    for (int i = 0; i < dst_norm_scaled.rows; i++)
    {
        for (int j = 0; j < dst_norm_scaled.cols; j++)
        {
            if (dst_norm_scaled.at<unsigned char>(i, j) >= minResponse)
            {
                hit = NMS(i, j, dst_norm_scaled, size);
                if(hit == true)
                {
                    hit = false;
                    logKeypoints.push_back(cv::KeyPoint(j, i, size));
                }
            }   
        }
    }

    windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, logKeypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
}

int main()
{
    cornernessHarris();
}