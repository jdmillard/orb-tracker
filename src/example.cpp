#include "example.h"

ExampleClass::ExampleClass()
{
  std::cout << "class instantiated" << std::endl;

  // initialize windows
  cv::namedWindow("frame", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("frame", 50, 50);

  cv::namedWindow("last keypoints", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("last keypoints", 50, 500);

  cv::namedWindow("search window", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("search window", 500 , 500);

  cv::namedWindow("matches", CV_WINDOW_AUTOSIZE);
  cv::moveWindow("matches", 50 , 700);

  // open video here
  video_path_ = "../videos/sample.mp4";
  cv::VideoCapture source_(video_path_);
  source_.set(CV_CAP_PROP_POS_FRAMES, 1);

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

  // generate the search window
  int expand = sss_ + 10;
  cv::Rect window_search = cv::Rect(col-expand, row-expand, w+(2*expand), h+(2*expand));
  roi_ = frame_(window_search).clone();

  // draw rectangle
  cv::rectangle(frame_, cv::Point(col,row), cv::Point(col+w, row+h), cv::Scalar(255, 0, 0), 2);
  // cv::imshow("frame", frame_);

  // ORB feature parameters
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

  orb_detector_->detect(roi_, keypoints_last_);

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
  source_.set(CV_CAP_PROP_POS_FRAMES, 2);
  if (!source_.isOpened())
    std::cout << "invalid video file path" << std::endl;

  while (true)
  {
    // get the next frame
    source_ >> frame_;
    if (frame_.empty())
      return;

    // generate the search window
    int expand = sss_ + 10;
    cv::Rect window_search = cv::Rect(col-expand, row-expand, w+(2*expand), h+(2*expand));
    cv::Mat roi_search = frame_(window_search).clone();

    // find features in the search window
    std::vector<cv::KeyPoint> keypoints_new;
    orb_detector_->detect(roi_search, keypoints_new);

    // generate descriptors
    cv::Mat descriptors_last, descriptors_new;
    orb_extractor_->compute(roi_, keypoints_last_, descriptors_last);
    orb_extractor_->compute(roi_search, keypoints_new, descriptors_new);

    // match the features
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true); // cv::NORM_HAMMING2 for when WTA_K==3 or 4
    std::vector<cv::DMatch> matches, good_matches;
    matcher.match(descriptors_last, descriptors_new, matches);

    // instead of keeping top %, keep the ones above a threshold
    // sort the matches by hamming distance
    std::sort(matches.begin(), matches.end());

    std::cout << "-------" << std::endl;

    std::vector<cv::KeyPoint> keypoints_keep;
    for (uint32_t i=0; i<matches.size(); i++)
    {
      // queryIdx is the id of the old keypoint
      // trainIdx is the id of the new keypoint
      std::cout << "---" << std::endl;
      std::cout << matches[i].distance << std::endl;
      std::cout << matches[i].queryIdx << std::endl; // index in vector of old keypoints
      std::cout << matches[i].trainIdx << std::endl; // index in vector of new keypoints
      if (true) // matches[i].distance<=4000
      {
        keypoints_keep.push_back(keypoints_new[matches[i].trainIdx]);
        good_matches.push_back(matches[i]);
      }

    }

    std::cout << matches.size() << std::endl;
    std::cout << good_matches.size() << std::endl;


    double x = 0;
    double y = 0;
    for (uint32_t i=0; i<keypoints_keep.size(); i++)
    {
      x += keypoints_keep[i].pt.x;
      y += keypoints_keep[i].pt.y;
    }

    // centroid of the points
    x = x/((double)keypoints_keep.size());// + col-expand;
    y = y/((double)keypoints_keep.size());// + row-expand;

    // find top-left corner of this position, given h, w
    x = x - (double)w/2;
    y = y - (double)h/2;

    // find global position
    x = x + col - expand;
    y = y + row - expand;

    // update window_ based on matched feature results
    col = x;
    row = y;
    window_ = cv::Rect(col, row, w, h);



    // all plotting
    cv::Mat frame = frame_.clone();
    cv::rectangle(frame, window_, cv::Scalar(255, 0, 0), 2);
    cv::imshow("frame", frame);

    // draw the last saved keypoints on their image
    cv::Mat roi = roi_.clone();
    cv::drawKeypoints(roi, keypoints_last_, roi);
    cv::imshow("last keypoints", roi);

    // draw the newly found keypoints on their image
    cv::Mat roi_search_draw = roi_search.clone();
    cv::drawKeypoints(roi_search_draw, keypoints_keep, roi_search_draw);
    cv::imshow("search window", roi_search_draw);

    // show the matches
    cv::Mat matches_draw;
    cv::drawMatches(roi_, keypoints_last_, roi_search, keypoints_new, matches, matches_draw);
    cv::imshow("matches", matches_draw);
    roi_ = roi_search.clone();
    keypoints_last_ = keypoints_keep;

    // plot the results of this iteration and wait for keypress
    auto key = cv::waitKey();
    if ((int)key==27)
      return;
  }
}


// need to consider when associations are way off from the rest
// so after convergence, there needs to be a check for results that are outside
// the cluster of the rest