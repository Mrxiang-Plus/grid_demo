//
// Created by xiang on 2019/10/15.
//

#ifndef GRIDDEMO_FEATURE_H
#define GRIDDEMO_FEATURE_H

#include "myslam/common_include.h"

//对传入的图片提取角点
void getPoints(cv::Mat image,std::vector<cv::Point2f> &points,cv::InputArray mask=cv::noArray(),int maxConer= -1);
//对传入的图片提取角点,重载，KeyPoint版
void getPoints(cv::Mat image,std::vector<cv::KeyPoint> &points,cv::InputArray mask=cv::noArray(),int maxConer= -1);
//按照grid提取角点
void getPoints_grid(cv::Mat image, int grid_size,int num_point_grid, std::vector<cv::Point2f> &points);
//按照grid提取角点,重载,加入总点数限制
void getPoints_grid(cv::Mat image, int grid_size,int num_point_grid, std::vector<cv::Point2f> &points, int maxConer);
//对取的点按照最小距离进行筛选
void minDistance(cv::Mat image, std::vector<cv::Point2f> &points, int minDistance=30,int maxCorners=1000);
//对取的点按照最小距离进行筛选,重载
void minDistance(cv::Mat image, std::vector<cv::KeyPoint> &points, int minDistance=30,int maxCorners=1000);
//设置mask区域
cv::Mat setmask(cv::Mat image,std::vector<cv::Point2f> points,double MIN_DIST=30);
//去除outlier
void rejectWithF(std::vector<cv::Point2f> &points1,std::vector<cv::Point2f> &points2,double F_THRESHOLD=1.0);
//用于定义点排序
//    bool compare(cv::KeyPoint a,cv::KeyPoint b);
//画出grid
void drawGrid(cv::Mat &image, int grid_size);
//去除追踪失败的点
void reducePoints(std::vector<cv::Point2f> &v, std::vector<uchar> status);
//去除追踪失败的点,按索引，从零开始
void reducePoints(std::vector<cv::Point2f> &v, int index );
//去除追踪失败点的追踪次数
void reducePoints(std::vector<int > &v, std::vector<uchar> status);
//画出轨迹
void drawTrace(std::vector<cv::Point2f> &points1,std::vector<cv::Point2f> &points2,cv::Mat image);
//得到当前点的深度
float getDepth(cv::Mat image_depth,cv::Point2f point,float depth_scale=1.0);

#endif //GRIDDEMO_FEATURE_H
