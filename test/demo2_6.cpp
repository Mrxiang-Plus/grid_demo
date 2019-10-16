//
// Created by xiang on 2019/10/10.
//
/*
 * 光流程序
 * 加入outlier去除
 * 加入grid,grid划分版本2
 * 加入补点操作
 * 更改为Shi-Tomasi角点
 */

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;

const int MAXCONER = 200;//最大角点数
int grid = 100;//grid的size
int num_point = 5;//每个grid的点数

cv::Mat this_image, prev_image;//图片信息
cv::Mat fast_image;
vector<cv::Point2f> points_prev, points_this;//存储采集到的角点,上一帧、当前帧
cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准
double F_THRESHOLD = 1.0;

//用于定义点排序
bool compare(cv::KeyPoint a,cv::KeyPoint b){
    return a.response > b.response;
}
//对传入的图片提取角点
void getPoints(cv::Mat image,vector<cv::Point2f> &points,cv::InputArray mask=cv::noArray(),int maxConer= -1){
    vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
    //GFTT检测器
    cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(100, 0.01,30,3, false,0.04);
    ShiDetector -> detect(image, points_Fast, mask);
    //按照响应值降序排序
    sort(points_Fast.begin(),points_Fast.end(),compare);

    /*决定导入点的数量
     * maxConer > 0,按照maxConer导入
     * maxConer = 0,不导入
     * maxConer < 0,无限制
     */

    if (maxConer > 0){
        for (int i = 0; i < maxConer && i< points_Fast.size(); ++i) {
            points.push_back(points_Fast[i].pt);
        }
    }
    if (maxConer < 0){
        for (auto kp:points_Fast)
            points.push_back(kp.pt);
    }

}
//对传入的图片提取角点,重载，KeyPoint版
void getPoints(cv::Mat image,vector<cv::KeyPoint> &points,cv::InputArray mask=cv::noArray(),int maxConer= -1){
    vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
    //GFTT检测器
    cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(100, 0.01,30,3, false,0.04);
    ShiDetector -> detect(image, points_Fast, mask);
    //按照响应值降序排序
    sort(points_Fast.begin(),points_Fast.end(),compare);

    /*决定导入点的数量
     * maxConer > 0,按照maxConer导入
     * maxConer = 0,不导入
     * maxConer < 0,无限制
     */

    if (maxConer > 0){
        for (int i = 0; i < maxConer; ++i) {
            points.push_back(points_Fast[i]);
        }
    }
    if (maxConer < 0){
        for (auto kp:points_Fast)
            points.push_back(kp);
    }

}
//按照grid提取角点
void getPoints_grid(cv::Mat image, int grid_size,int num_point_grid, vector<cv::Point2f> &points){
    int width = image.cols;
    int height = image.rows;
    int n = width / grid_size; // 一行的grid数；
    int m = height / grid_size; // 一列的grid数
    //每个grid检测
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
            //每块grid检测，并选出最佳5个
            mask.colRange(grid_size*i,grid_size*(i+1)).rowRange(grid_size*j,grid_size*(j+1)).setTo(255);
            getPoints(image,points,mask,num_point_grid);
        }
    }
    //grid之外的边缘区域,列边缘，行边缘
    for (int k = 0; k < n; ++k) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*k ,grid_size*(k+1)).rowRange(grid_size*m,height).setTo(255);
        getPoints(image,points,mask,num_point_grid);
    }
    for (int l = 0; l < m; ++l) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*n ,width).rowRange(grid_size*l,grid_size*(l+1)).setTo(255);
        getPoints(image,points,mask,num_point_grid);
    }

    cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
    mask.colRange(grid_size*n ,width).rowRange(grid_size*m,height).setTo(255);
    getPoints(image,points,mask,num_point_grid);
}
//按照grid提取角点,重载,加入总点数限制
void getPoints_grid(cv::Mat image, int grid_size,int num_point_grid, vector<cv::Point2f> &points, int maxConer){
    int width = image.cols;
    int height = image.rows;
    int n = width / grid_size; // 一行的grid数；
    int m = height / grid_size; // 一列的grid数
    vector<cv::KeyPoint> points_temp;
    //每个grid检测
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
            //每块grid检测，并选出最佳5个
            mask.colRange(grid_size*i,grid_size*(i+1)).rowRange(grid_size*j,grid_size*(j+1)).setTo(255);
            getPoints(image,points_temp,mask,num_point_grid);
        }
    }
    //grid之外的边缘区域,列边缘，行边缘
    for (int k = 0; k < n; ++k) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*k ,grid_size*(k+1)).rowRange(grid_size*m,height).setTo(255);
        getPoints(image,points_temp,mask,num_point_grid);
    }
    for (int l = 0; l < m; ++l) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*n ,width).rowRange(grid_size*l,grid_size*(l+1)).setTo(255);
        getPoints(image,points_temp,mask,num_point_grid);
    }

    cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
    mask.colRange(grid_size*n ,width).rowRange(grid_size*m,height).setTo(255);
    getPoints(image,points_temp,mask,num_point_grid);

    sort(points_temp.begin(),points_temp.end(),compare);
    if (maxConer > 0){
        for (int i = 0; i<maxConer && i<points_temp.size(); ++i) {
            points.push_back(points_temp[i].pt);
        }
    }
    if (maxConer < 0){
        for (auto kp:points_temp)
            points.push_back(kp.pt);
    }

}
//画出grid
void drawGrid(cv::Mat &image, int grid_size){
    int width = image.cols;
    int height = image.rows;
    int n = width / grid_size; // 一行的grid数；
    int m = height / grid_size; // 一列的grid数
    //画出grid线条
    for (int i = 1; i <= n; ++i) {
        cv::line(image,cv::Point2f(grid_size*i,0),cv::Point2f(grid_size*i,height),cv::Scalar(255,255,255));
    }

    for (int i = 1; i <= m; ++i) {
        cv::line(image,cv::Point2f(0,grid_size*i),cv::Point2f(width,grid_size*i),cv::Scalar(255,255,255));
    }
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


        //第一帧提取角点
        if (isfrist){
            double start_frist = cv::getTickCount();
            getPoints_grid(this_image,grid,num_point,points_this);
            double time_frsit = (cv::getTickCount() - start_frist) / (double)cv::getTickFrequency();
            cout<<"time for frist image: "<<time_frsit<<endl;
            drawGrid(fast_image,grid);

            //在当前帧上画出角点
            for (int i = 0; i < points_this.size(); ++i) {
                cv::circle(fast_image,points_this[i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }
            //亚像素角点精确化
            cv::cornerSubPix(this_image,points_this,cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            isfrist = false;
        }
            //后续进行光流追踪
        else{
            vector<uchar > status_Fast;
            vector<float > err_Fast;
            double start_flow = cv::getTickCount();
            cv::calcOpticalFlowPyrLK(prev_image,this_image,points_prev,points_this,status_Fast,err_Fast,cv::Size(21,21),3,termCriteria,0,0.001);
            double time_flow = (cv::getTickCount() - start_flow) / cv::getTickFrequency();
            cout << "time for flow" << time_flow<<endl;
            reducePoints(points_this,status_Fast);
            reducePoints(points_prev,status_Fast);
            rejectWithF(points_prev,points_this);//去除outlier

            //补点操作
            int num_add = MAXCONER - points_this.size();
            if(num_add > 0){
                double start_add = cv::getTickCount();
                getPoints_grid(this_image,grid,num_point,points_this,num_add);
                double time_add = (cv::getTickCount() - start_add) / cv::getTickFrequency();
                cout <<"time for add point: "<< time_add<<endl;
            }

            if(points_this.size() == 0) {
                cout<<"all points are loss"<<endl;
                return 0;
            }
            //绘制点
            for (int i = 0; i < points_this.size(); ++i) {
                cv::circle(fast_image, points_this[i], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
            }
            //drawTrace(points_prev,points_this,fast_image);

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



