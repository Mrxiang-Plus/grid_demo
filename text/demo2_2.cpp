//
// Created by xiang on 2019/10/9.
//
/*
 * 加入outlier去除的光流程序
 * 无grid版
 */

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;


cv::Mat this_image, prev_image;//图片信息
cv::Mat fast_image;
vector<cv::Point2f> points_prev, points_this;//存储采集到的角点,上一帧、当前帧
double F_THRESHOLD = 1.0;
//对传入的图片提取角点
void getPoints(cv::Mat image,vector<cv::Point2f> &points ){
    vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
    //Fast检测器
    cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create(20, true);
    FastDetector -> detect(image, points_Fast );
    // cv::KeyPointsFilter::retainBest(points_Fast,300);//取前300个特征点
    // cv::cornerSubPix(this_image,points[1],cv::Size(10,10),cv::Size(-1,-1),termCriteria);
    //点导入
    for(auto kp:points_Fast)
        points.push_back(kp.pt);
}
//去除追踪失败的点
void reducePoints(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
//画出轨迹
void drawTrace(vector<cv::Point2f> &points1,vector<cv::Point2f> &points2,cv::Mat image){
    if(points1.size() != points2.size()) cout<<"wrong point number,can't draw trace"<<endl;
    else
        for (int i = 0; i < points1.size(); ++i) {
            cv::line(image, points1[i], points2[i], cv::Scalar(255, 255, 255), 2, 8);
            cv::circle(image, points2[i], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
        }
}

//去除outlier
void rejectWithF(vector<cv::Point2f> &points1,vector<cv::Point2f> &points2){
    vector<uchar> status;
    if(points2.size() >= 8){
        cv::findFundamentalMat(points1,points2,cv::FM_RANSAC,F_THRESHOLD,0.99,status);
        reducePoints(points1,status);
        reducePoints(points2,status);

    }
}

int main(int argc, char** argv){
    if(argc != 2){
        cout<<"wrong input\nusage: grid path_to_dataset"<<endl;
        return 1 ;
    }

    //一些必要的数据
    cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准

    string dataset_path = argv[1];
    string dataset = dataset_path + "/file.txt";
    bool isfrist = true;
    int num = 0;
    ifstream fin(dataset);//读取数据集文件


    ///读取数据并进行处理
    while(true){
        string picture_file;//存储当前图片地址

        fin >> picture_file;
        if (fin.eof())   break;

        this_image = cv::imread(picture_file);
        this_image.copyTo(fast_image);
        cv::cvtColor(this_image, this_image, cv::COLOR_BGR2GRAY);

//        cv::imshow("Fast",this_image);
//        cv::waitKey(0);
        //第一帧提取角点
        if (isfrist){
            getPoints(this_image,points_this);

            //在当前帧上画出角点
            for (int i = 0; i < points_this.size(); ++i) {
                cv::circle(fast_image,points_this[i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }

            isfrist = false;
        }
            //后续进行光流追踪
        else{
            vector<uchar > status_Fast;
            vector<float > err_Fast;
            cv::calcOpticalFlowPyrLK(prev_image,this_image,points_prev,points_this,status_Fast,err_Fast,cv::Size(10,10),3,termCriteria,0,0.001);
            reducePoints(points_this,status_Fast);
            reducePoints(points_prev,status_Fast);
            rejectWithF(points_prev,points_this);//去除outlier

            if(points_this.size() == 0) {
                cout<<"all points are loss"<<endl;
                break;
            }

            drawTrace(points_prev,points_this,fast_image);
        }


        cout<<"num of points of Fast : "<<points_this.size()<<endl;
        cv::imshow("Fast",fast_image);

        //进行帧迭代
        swap(points_this,points_prev);
        swap(this_image,prev_image)  ;
        num++;
        cout<<"num of image:"<<num<<endl;
        cv::waitKey(-1);
    }
    return 0;
}
