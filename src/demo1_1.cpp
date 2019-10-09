//
// Created by xiang on 2019/9/19.
//
///使用特定的图片
///测试使用grid

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;

int main(int argv, char ** argc){
    string image_path = "..\\data\\basketball1.png";
    cv::Mat image ,image_gray;

    image = cv::imread(image_path);//读取图片
    cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);//转化为灰度图

    //划分grid
    int grid = 30;
    int width = image.cols;
    int height = image.rows;
    int n = width / grid; // 一行的grid数；
    int m = height / grid; // 一列的grid数
    int num_points = 0;//总的角点数
    //Fast特征检测器
    vector<cv::KeyPoint> points_fast;
    cv::namedWindow("Fast",cv::WINDOW_NORMAL);
    cv::resizeWindow("Fast",1080,720);

    cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(40,true,cv::FastFeatureDetector::TYPE_9_16);

    double start_Fast = cv::getTickCount();// 检测开始时
    //每个grid检测
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            //每块grid检测，并选出最佳5个
            fastDetector -> detect(image_gray.colRange(grid*i,grid*(i+1)).rowRange(grid*j,grid*(j+1)),points_fast);
            cv::KeyPointsFilter::retainBest(points_fast,5);
            //角点坐标有相对与grid调整为相对与完整图片
            for (int k = 0; k < points_fast.size(); ++k) {
                points_fast[k].pt += cv::Point2f(grid*i,grid*j);
            }
            cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);//在原图上画出角点
            num_points += points_fast.size();
        }
    }
    //grid之外的边缘区域,列边缘，行边缘
    fastDetector -> detect(image_gray.colRange(grid*n ,width),points_fast);
    for (int k = 0; k < points_fast.size(); ++k) {
        points_fast[k].pt += cv::Point2f(grid*n,grid*m);
    }
    cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);

    fastDetector -> detect(image_gray.colRange(0,grid*n).rowRange(grid*m,height),points_fast);
    for (int k = 0; k < points_fast.size(); ++k) {
        points_fast[k].pt += cv::Point2f(grid*n,grid*m);
    }
    cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);

    //画出grid线条
    for (int i = 1; i <= n; ++i) {
        cv::line(image,cv::Point2f(grid*i,0),cv::Point2f(grid*i,grid*m),cv::Scalar(255,255,255));
    }

    for (int i = 1; i <= m; ++i) {
        cv::line(image,cv::Point2f(0,grid*i),cv::Point2f(grid*n,grid*i),cv::Scalar(255,255,255));
    }

    double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间

    cv::imshow("Fast", image);
    cout << "fast point number: " << num_points << "\ntime for fast: " << time_Fast << " s\n" << endl;

    cv::waitKey(-1);
    return 0;
}
