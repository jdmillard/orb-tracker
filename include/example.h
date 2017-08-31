#include <iostream>
#include <math.h>
#include <chrono>
#include <opencv2/opencv.hpp>
// not sure if these two are needed:
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


class ExampleClass
{
public:
  ExampleClass();
  void operations();

private:
  std::string video_path_;
  cv::VideoCapture source_;

  cv::Mat frame_;

  cv::Mat roi_;
  cv::Rect window_;

  cv::TermCriteria criteria_;
  uint32_t col, row, w, h;
  int sss_;

  cv::Ptr<cv::FeatureDetector> orb_detector_;
  cv::Ptr<cv::DescriptorExtractor> orb_extractor_;

  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;

};