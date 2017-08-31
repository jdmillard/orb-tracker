#include "example.h"

ExampleClass::ExampleClass()
{
  std::cout << "class instantiated" << std::endl;

  // initialize windows
  cv::namedWindow("frame", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("frame", 50, 50);

  // open video here
  video_path_ = "../videos/sample.mp4";
  cv::VideoCapture source_(video_path_);

  if (!source_.isOpened())
    std::cout << "invalid video file path" << std::endl;

  source_ >> frame_;
  if(frame_.empty())
  {
    std::cout << "frame is bad" << std::endl;
    return;
  }

  // hard-coded location of object in first frame
  col = 160;
  row = 160;
  w   = 40;
  h   = 60;
  window_ = cv::Rect(col, row, w, h);
  roi_ = frame_(window_).clone();
  cv::imshow("roi", roi_);

  int sss = 31; // <<<<<<<<<<<<<<<<<<<<<<<< PATCH SIZE <<<<<<<<<<<<<<<<<<<<<<<<
  cv::Rect window_aug = cv::Rect(col-sss, row-sss, w+(2*sss), h+(2*sss));
  cv::Mat roi2 = frame_(window_aug).clone();
  cv::imshow("augmented", roi2);

  // draw rectangle
  cv::rectangle(frame_, cv::Point(col,row), cv::Point(col+w, row+h), cv::Scalar(255, 0, 0), 2);
  cv::imshow("frame", frame_);

  // use roi_ to extract ORB features
  std::vector<cv::KeyPoint> keypoints;
  int nfeatures     = 500;                    // max number of features (500)
  float scaleFactor = 1.2;                    // pyramid scaling: 2 is poorer matching, 1 is more computation
  int nlevels       = 8;                      // number of pyramid levels
  int edgeThreshold = sss;                    // size of border where features are not detected
  int firstLevel    = 0;                      //
  int WTA_K         = 2;                      // points used in each element of descriptor
  int scoreType     = cv::ORB::HARRIS_SCORE;  // used to rank cv::ORB::FAST_SCORE is less stable, but faster
  int patchSize     = sss;                    // size of descriptor patch
  int fastThreshold = 20;                     //

  cv::Ptr<cv::FeatureDetector> orb_detector = cv::ORB::create(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize,
    fastThreshold);
  orb_detector->detect(roi2, keypoints);

  std::cout << keypoints.size() << std::endl;

  cv::Mat descriptors;
  cv::Ptr<cv::DescriptorExtractor> orb_extractor = cv::ORB::create(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize,
    fastThreshold);
  orb_extractor->compute(roi2, keypoints, descriptors);

  std::cout << descriptors.size() << std::endl;



  cv::drawKeypoints(roi2, keypoints, roi2);
  cv::imshow("keypoints", roi2);

  // close the video source for scoping reasons
  source_.release();
}

// ----------------------------------------------------------------------------

void ExampleClass::operations()
{
  // perform operations
  std::cout << "starting operations; press esc to end early" << std::endl;

  // reopen the video source and resume at frame 2
  source_.open(video_path_);
  source_.set(CV_CAP_PROP_POS_FRAMES, 1);
  if (!source_.isOpened())
    std::cout << "invalid video file path" << std::endl;

  while (true)
  {
    // plot the results of the last iteration and wait for keypress
    auto key = cv::waitKey();
    if ((int)key==27)
      return;

    // get the next frame
    source_ >> frame_;
    if (frame_.empty())
      return;

    // match orb in this frame

    // update window_ based on matched features
    col = 165;
    row = 165;
    window_ = cv::Rect(col, row, w, h);

    // update roi display
    roi_ = frame_(window_).clone();
    cv::imshow("roi", roi_);

    // draw rectangle
    cv::rectangle(frame_, window_, cv::Scalar(255, 0, 0), 2);
    cv::imshow("frame", frame_);

    // NOTE: this operates on the entire window, use the current estimate
    // to select a subwindow, decreasing the amount of backprop required?
  }
}