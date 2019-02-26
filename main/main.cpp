#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <vector>
#include "kcftracker.hpp"
#include <opencv2/core.hpp>

#define test_fps

using namespace cv;
using namespace std;
using namespace dsst;

int main()
{
    //***********declare fundamental variable**********//
    double fps;
    double time_used;
    cv::Mat frame;
    cv::Mat frame_gray;
    cv::Rect2d roi;
    VideoCapture cap("/home/ubutnu/Video/s_video/h3.MP4");
    int frame_count;    //number of passed frame

 #ifdef test_fps
    double start_fps,end_fps,dur_fps;
#endif

    //*****************declare tracker****************//
    bool hog=true;
    bool fixed_window=false;
    bool multiscale=true;
    bool lab=false;
    bool dsst =true;
    DSSTTracker tracker(hog);

    //****************initialize tracker*****************//
    cap >> frame;
    if(frame.empty() == true)
    {
        std::cout<<"Video is empty!"<<std::endl;
        return 0;
    }
    namedWindow("frame");
    imshow("frame", frame);

    roi = selectROI("frame", frame);

//    cvtColor(frame,frame_gray,COLOR_RGB2GRAY,0);
//    tracker -> init(roi,frame_gray);
    tracker.init(frame,roi);
    for(frame_count = 1;;frame_count++)
    {
#ifdef test_fps
        start_fps = clock();
#endif
        cap>>frame;
//        cvtColor(frame,frame_gray,COLOR_RGB2GRAY,0);
        if(frame.empty() == true)
        {
            std::cout<<"Video Over!"<<std::endl;
            break;
        }
        tracker.update(frame,roi);
#ifdef test_fps
        end_fps = clock();
        dur_fps =end_fps-start_fps;
        time_used = (double)(dur_fps)/CLOCKS_PER_SEC;
        fps =1/time_used;
        time_used *= 1000;
        cout<<"fps/time used:"<<fps<<"/"<<time_used<<"ms"<<endl;
#endif
        rectangle(frame,roi,Scalar(255,0,0),2);
        imshow("frame",frame);
        if(waitKey(30)>0)
            break;
    }
}
