//
// Created by xiang on 2019/10/17.
//
#include "myslam/camera.h"
using namespace std;


void getParameter(){
    cv::FileStorage fs("../config/default.yaml",cv::FileStorage::READ);
    if (!fs.isOpened()){
        cout <<"No file!" << endl;
        return;
    }
    fx = (double)fs["camera.fx"];
    fy = (double)fs["camera.fy"];
    cx = (double)fs["camera.cx"];
    cy = (double)fs["camera.cy"];
    depth_scale = (double)fs["camera.depth_scale"];
    dataset_dir = (string)fs["dataset_dir"];
    grid_size = (int)fs["grid_size"];
    points_num_grid = (int)fs["points_num_grid"];
    F_THRESHOLD = (double)fs["F_THRESHOLD"];
    MAXCONER = (int)fs["MAXCONER"];
    MIN_DIST = (int)fs["MIN_DIST"];
    min_inliers = (int)fs["min_inliers"];
}

cv::Point3f pixel2camera (cv::Point2f p_p, float depth){
    return cv::Point3f(
            (p_p.x - cx) * depth / fx,
            (p_p.y - cy) * depth / fy,
            depth);
}

cv::Point2f camera2pixel (cv::Point3f p_c){
    return cv::Point2f(
            p_c.x * fx / p_c.z + cx,
            p_c.y * fy / p_c.z + cy);
}

cv::Point3f world2camera ( cv::Point3f p_w, cv::Matx44d T_c_w )
{
    cv::Matx41d point(p_w.x, p_w.y, p_w.z,1);
    cv::Matx41d p_c = T_c_w * point;
    return cv::Point3f(p_c(0,0),p_c(1,0),p_c(2,0));
}

cv::Point3f camera2world ( cv::Point3f p_c, cv::Matx44d T_c_w ){
    cv::Matx44d T_w_c = T_c_w.inv();
    cv::Matx41d point(p_c.x, p_c.y, p_c.z,1);
    cv::Matx41d p_w = T_w_c * point;
    return cv::Point3f(p_w(0,0),p_w(1,0),p_w(2,0));
}

cv::Point3f pixel2world(cv::Point2f p_p, float depth, cv::Matx44d T_c_w){
    return camera2world(pixel2camera(p_p,depth),T_c_w);
}

cv::Point2f world2pixel(cv::Point3f p_w,cv::Matx44d T_c_w){
    return camera2pixel(world2camera(p_w,T_c_w));
}

void getPose(vector<cv::Point3f> points_3d,vector<cv::Point2f> points_2d,cv::Mat &rvec,cv::Mat &tvec,cv::Mat &points_true){
    if(points_3d.size() < 8){
        cout<<"点数过少，无法pnp求取位姿"<<endl;
    }
    cv::Mat K = (cv::Mat_<double >(3,3)<<
                                       fx,0,cx,
            0,fy,cy,
            0,0,1);
    solvePnPRansac(points_3d,points_2d,K,cv::noArray(),rvec,tvec, false,100, 4.0, 0.99,points_true);
}
