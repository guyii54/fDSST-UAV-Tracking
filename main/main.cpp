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

#define TEST_FPS
//#define SAVE_VIDEO


using namespace cv;
using namespace std;
using namespace dsst;

int main()
{
    //***********declare fundamental variable**********//
    double fps;
    double time_used;
    cv::Mat frame;
//    cv::Mat frame_gray;
    cv::Rect2d roi;
    std::string read_path = "/home/nvidia/Videos/s_video/";
    std::string filename;
    std::string suffix = ".mp4";
    std::cin>>filename;

//    std::string read_path = "/home/ubuntu/Videos/";
//    std::string filename;
//    std::string suffix = "/000%03d.jpg";
//    std::cin >> filename;

//    filename = "longtime1";
    std::string w_read_path = read_path+filename+suffix;
    VideoCapture cap(w_read_path);
    int frame_count;    //number of passed frame

#ifdef SAVE_VIDEO
    std::string write_path="../processed/";
    std::string w_write_path = write_path+filename+".avi";
    VideoWriter writer(w_write_path,CV_FOURCC('M','J','P','G'),24,Size(1280,720));
#endif

 #ifdef TEST_FPS
    double start_fps,end_fps,dur_fps;
#endif



    //*****************declare tracker****************//
    bool hog=true;
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


		cap>>frame;

#ifdef TEST_FPS
        start_fps = clock();
#endif

        

//        cvtColor(frame,frame_gray,COLOR_RGB2GRAY,0);
        if(frame.empty() == true)
        {
            std::cout<<"Video Over!"<<std::endl;
            break;
        }
        tracker.update(frame,roi);


#ifdef TEST_FPS
        end_fps = clock();
        dur_fps =end_fps-start_fps;
        time_used = (double)(dur_fps)/CLOCKS_PER_SEC;
        fps =1/time_used;
        time_used *= 1000;
		cout<<"roi size:"<<roi.width<<"*"<<roi.height<<endl;
        cout<<"fps/time used:"<<fps<<"/"<<time_used<<"ms"<<endl;
        cout<<"-----------new frame----------"<<endl;
#endif

//        rectangle(frame,tracker.extracted_roi,Scalar(0,0,255),2);
        rectangle(frame,roi,Scalar(255,0,0),2);
        imshow("frame",frame);


#ifdef SAVE_VIDEO
        writer << frame;
#endif
        if(waitKey(30)>0)
            break;
    }
}
