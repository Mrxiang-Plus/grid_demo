//
// Created by xiang on 2019/10/15.
//

/*
 *采用vins思想补点
 * fast角点
 * 加入最小距离限制
 * 加入位姿计算
 */


#include "myslam/feature.h"
using namespace std;
const int MAXCONER = 200;//最大角点数
const int MIN_DIST = 30;//点间距
int grid = 100;//grid的size
int num_point = 5;//每个grid的点数
cv::Mat this_image, prev_image,this_depth;//图片信息
cv::Mat fast_image;

vector<cv::Point2f> points_prev, points_this;//存储采集到的角点,上一帧、当前帧
cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准
double F_THRESHOLD = 1.0;
double fx=0,fy=0,cx=0,cy=0,depth_scale=0;
string dataset_dir;

//读取配置文件
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
}
//像素坐标转换为相机坐标
cv::Point3d pixel2camera (cv::Point2f point, double depth){
    return cv::Point3f(
            (point.x - cx) * depth/fx,
            (point.y - cy) * depth/fy,
            depth);
}
//3D——2D 求解位姿
void getPose(vector<cv::Point3f> points_3d,vector<cv::Point2f> points_2d,cv::Mat &rvec,cv::Mat &tvec){
    cv::Mat K = (cv::Mat_<double >(3,3)<<
            fx,0,cx,
            0,fy,cy,
            0,0,1);
    solvePnP(points_3d,points_2d,K,cv::noArray(),rvec,tvec, false,cv::SOLVEPNP_ITERATIVE);
}

int main(int argc, char** argv){
    if(argc != 1){
        cout<<"wrong input\nusage: grid "<<endl;
        return 1 ;
    }
    getParameter();
    cout<<"fx :"<< fx<<" fy :"<<fy<<" cx : "<<cx<<" cy : "<<cy<<" depth_scale : "<<depth_scale<<endl;
    //一些必要的数据
    cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);//创建一个显示窗口

    string dataset = dataset_dir + "/associate.txt";
    bool isfrist = true;
    int num = 0;
    ifstream fin(dataset);//读取数据集文件
    cv::Mat rvec,tvec;
    ///读取数据并进行处理
    while(true){
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_file = dataset_dir + "/" + rgb_file;
        depth_file = dataset_dir + "/" + depth_file;
        if (fin.eof())   break;

        vector<cv::Point3f> points_3d;//当前点的相机坐标
        this_image = cv::imread(rgb_file);
        this_image.copyTo(fast_image);
        this_depth = cv::imread(depth_file);
        cv::cvtColor(this_image, this_image, cv::COLOR_BGR2GRAY);

        //第一帧提取角点
        if (isfrist){
            double start_frist = cv::getTickCount();
            getPoints_grid(this_image,grid,num_point,points_this);
            double time_frsit = (cv::getTickCount() - start_frist) / (double)cv::getTickFrequency();
            drawGrid(fast_image,grid);

            //在当前帧上画出角点
            for (int i = 0; i < points_this.size(); ++i) {
                cv::circle(fast_image,points_this[i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }
            //亚像素角点精确化
            cv::cornerSubPix(this_image,points_this,cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            isfrist = false;
            cout<<"time for frist image: "<<time_frsit<<endl;
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
            rejectWithF(points_prev,points_this,F_THRESHOLD);//去除outlier
            //求解两帧之间位姿
            for (int i = 0; i < points_this.size(); ++i) {
                double depth = getDepth(this_depth,points_prev[i],depth_scale);
                points_3d.push_back(pixel2camera(points_prev[i],depth));
            }
            double start_getpose = cv::getTickCount();
            getPose(points_3d,points_this,rvec,tvec);
            double time_getpose = (cv::getTickCount() - start_getpose) / cv::getTickFrequency();
            cout <<"time for get pose: "<< time_getpose<<endl;

            //补点操作
            int num_add = MAXCONER - points_this.size();
            if(num_add > 0){
                cv::Mat mask = setmask(this_image,points_this,MIN_DIST);
                double start_add = cv::getTickCount();
                getPoints(this_image,points_this,mask,num_add);
                double time_add = (cv::getTickCount() - start_add) / cv::getTickFrequency();
                cout <<"time for add point: "<< time_add<<endl;
//                cv::imshow("mask",mask);
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
        points_3d.clear();
        //进行帧迭代
        swap(points_this,points_prev);
        swap(this_image,prev_image)  ;
        num++;
        cout<<"num of image:"<<num<<endl;
        cv::waitKey(-1);
    }
    return 0;
}
