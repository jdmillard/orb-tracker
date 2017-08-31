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

    // for (uint32_t i=0; i<keypoints_new.size(); i++)
    // {
    //   std::cout << "---" << std::endl;
    //   std::cout << i << std::endl;
    //   std::cout << keypoints_new[i].class_id << std::endl;
    //   std::cout << keypoints_new[i].angle << std::endl;
    //   std::cout << keypoints_new[i].octave << std::endl;
    //   std::cout << keypoints_new[i].pt << std::endl;
    //   std::cout << keypoints_new[i].response << std::endl;
    //   std::cout << keypoints_new[i].size << std::endl;
    // }

    // generate descriptors
    cv::Mat descriptors_new;
    orb_extractor_->compute(search_window, keypoints_new, descriptors_new);

    // show the orb features in the search window
    cv::Mat temp_draw = search_window.clone();
    cv::drawKeypoints(search_window, keypoints_new, temp_draw);
    cv::imshow("search window", temp_draw);

    // match the features
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true); // cv::NORM_HAMMING2 for when WTA_K==3 or 4
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_, descriptors_new, matches);

    std::cout << matches.size() << std::endl;

    // show the feature pairs
    cv::Mat out;
    cv::drawMatches(roi_, keypoints_, search_window, keypoints_new, matches, out);
    cv::imshow("pairs", out);

    // std::cout << "------------" << std::endl;
    float size = matches.size();
    float percent = 0.20; // keep top 20% of best matches
    float des_length = size*percent;

    // sort the matches by distance
    std::sort(matches.begin(), matches.end());

    std::vector<cv::DMatch> matches2;

    // keep top %
    for (uint32_t i=0; i<des_length; i++)
      matches2.push_back(matches[i]);


    std::vector<cv::KeyPoint> keypoints_keep;
    for (uint32_t i=0; i<matches2.size(); i++)
    {
      // std::cout << "---" << std::endl;
      // std::cout << matches2[i].distance << std::endl;
      // std::cout << matches2[i].imgIdx << std::endl; // always zero
      // std::cout << matches2[i].queryIdx << std::endl;
      // std::cout << matches2[i].trainIdx << std::endl;

      // queryIdx is the id of the new keypoint
      // trainIdx is the id of the old keypoint
      keypoints_keep.push_back(keypoints_new[matches2[i].queryIdx]);
    }

    // show the kept orb features in the search window
    cv::Mat temp_draw2 = search_window.clone();
    cv::drawKeypoints(search_window, keypoints_keep, temp_draw2);
    cv::imshow("kept", temp_draw2);

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


    // refresh descriptors of the matched ones

    // "converge" on the set of descriptors based on movement
    // use the fact that there is movement to come up with the
    // set of keypoints that are consistently on the target, but regenerate
    // the descriptors for them when necessary. artifact tracks won't have
    // time to converge - hopefully





    // update window_ based on matched feature results
    col = x;
    row = y;
    window_ = cv::Rect(col, row, w, h);

    // update roi display
    // roi_ = frame_(window_).clone();
    // cv::imshow("roi", roi_);

    // draw rectangle
    cv::rectangle(frame_, window_, cv::Scalar(255, 0, 0), 2);
    cv::imshow("frame", frame_);

    // NOTE: this operates on the entire window, use the current estimate
    // to select a subwindow, decreasing the amount of backprop required?

    // plot the results of this iteration and wait for keypress
    auto key = cv::waitKey();
    if ((int)key==27)
      return;
  }
}