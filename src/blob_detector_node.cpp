#include <ros/ros.h>
#include "blob_detector/blob_detector.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "blob_detector");
  blob_detector::BlobDetector blob;
  ros::spin();
  return 0;
}

