#include "example.h"

ExampleClass::ExampleClass()
{
  std::cout << "class instantiated" << std::endl;

  // initialize windows
  cv::namedWindow("frame", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("frame", 50, 50);

  cv::namedWindow("original keypoints", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("original keypoints", 50, 500);

  cv::namedWindow("search window", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("search window", 300 , 500);

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

  sss_ = 31; // <<<<<<<<<<<<<<<<<<<<<<<< PATCH SIZE <<<<<<<<<<<<<<<<<<<<<<<<
  cv::Rect window_aug = cv::Rect(col-sss_, row-sss_, w+(2*sss_), h+(2*sss_));
  roi_ = frame_(window_aug).clone();

  // draw rectangle
  cv::rectangle(frame_, cv::Point(col,row), cv::Point(col+w, row+h), cv::Scalar(255, 0, 0), 2);
  cv::imshow("frame", frame_);

  // use roi_ to extract ORB features
  int nfeatures     = 500;                    // max number of features
  float scaleFactor = 1.2;                    // pyramid scaling: 2 is poorer matching, 1 is more computation
  int nlevels       = 8;                      // number of pyramid levels
  int edgeThreshold = sss_;                   // size of border where features are not detected
  int firstLevel    = 0;                      //
  int WTA_K         = 2;                      // points used in each element of descriptor
  int scoreType     = cv::ORB::HARRIS_SCORE;  // used to rank cv::ORB::FAST_SCORE is less stable, but faster
  int patchSize     = sss_;                   // size of descriptor patch
  int fastThreshold = 20;                     //

  orb_detector_ = cv::ORB::create(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize,
    fastThreshold);
  orb_detector_->detect(roi_, keypoints_);

  std::cout << keypoints_.size() << std::endl;

  orb_extractor_ = cv::ORB::create(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize,
    fastThreshold);
  orb_extractor_->compute(roi_, keypoints_, descriptors_);

  std::cout << descriptors_.size() << std::endl;



  cv::drawKeypoints(roi_, keypoints_, roi_);
  cv::imshow("original keypoints", roi_);

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

    // generate the search window
    int expand = sss_ + 10;
    cv::Rect window_search = cv::Rect(col-expand, row-expand, w+(2*expand), h+(2*expand));
    cv::Mat search_window = frame_(window_search).clone();

    // find features in the search window
    std::vector<cv::KeyPoint> keypoints_new;
    orb_detector_->detect(search_window, keypoints_new);

    // generate descriptors
    cv::Mat descriptors_new;
    orb_extractor_->compute(search_window, keypoints_new, descriptors_new);

    // show the orb features in the search window
    cv::drawKeypoints(search_window, keypoints_new, search_window);
    cv::imshow("search window", search_window);

    // match the features
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true); // cv::NORM_HAMMING2 for when WTA_K==3 or 4
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_, descriptors_new, matches);

    std::cout << matches.size() << std::endl;

    // show the feature pairs
    cv::Mat out;
    cv::drawMatches(roi_, keypoints_, search_window, keypoints_new, matches, out);
    cv::imshow("pairs", out);





    // update window_ based on matched feature results
    col = 165;
    row = 165;
    window_ = cv::Rect(col, row, w, h);

    // update roi display
    // roi_ = frame_(window_).clone();
    // cv::imshow("roi", roi_);

    // draw rectangle
    cv::rectangle(frame_, window_, cv::Scalar(255, 0, 0), 2);
    cv::imshow("frame", frame_);

    // NOTE: this operates on the entire window, use the current estimate
    // to select a subwindow, decreasing the amount of backprop required?
  }
}