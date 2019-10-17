//
// Created by xiang on 2019/10/17.
//

#ifndef GRIDDEMO_CAMERA_H
#define GRIDDEMO_CAMERA_H

#include "myslam/common_include.h"
double fx=0,fy=0,cx=0,cy=0,depth_scale=0;//相机内参
std::string dataset_dir;//数据集位置
int grid_size = 30;//grid的尺寸
int points_num_grid = 5;//每个grid的最大取点数
double F_THRESHOLD = 1.0;//去outlier要用的参数
int MAXCONER = 200;//每帧的最大角点数
int MIN_DIST = 30;//点之间的最小距离
int min_inliers = 30;//两帧之间最小内联点

//读取配置文件
void getParameter();
//像素坐标转换为相机坐标
cv::Point3f pixel2camera (cv::Point2f p_p, float depth);
//相机坐标转换为像素坐标
cv::Point2f camera2pixel (cv::Point3f p_c);
//世界坐标转化为相机坐标
cv::Point3f world2camera ( cv::Point3f p_w, cv::Matx44d T_c_w );
//相机坐标转化为世界坐标
cv::Point3f camera2world ( cv::Point3f p_c, cv::Matx44d T_c_w );
//像素坐标转化为世界坐标
cv::Point3f pixel2world(cv::Point2f p_p, float depth, cv::Matx44d T_c_w);
//世界坐标转化为像素坐标
cv::Point2f world2pixel(cv::Point3f p_w,cv::Matx44d T_c_w);
//3D——2D 求解位姿
void getPose(std::vector<cv::Point3f> points_3d,std::vector<cv::Point2f> points_2d,cv::Mat &rvec,cv::Mat &tvec,cv::Mat &points_true);

#endif //GRIDDEMO_CAMERA_H
