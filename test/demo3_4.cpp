//
// Created by xiang on 2019/10/18.
//


/*
 *更换补点方式
 *采用vins思想
 * fast角点
 * 加入最小距离限制
 * 结构优化
 */


#include "myslam/camera.h"
#include "myslam/feature.h"
using namespace std;

int main(int argc, char** argv){
    if(argc != 1){
        cout<<"wrong input\nusage: grid "<<endl;
        return 1 ;
    }
    getParameter();

    cout<<"fx :"<< fx<<" fy :"<<fy<<" cx : "<<cx<<" cy : "<<cy<<" depth_scale : "<<depth_scale<<endl;
    cv::Mat K = (cv::Mat_<double >(3,3)<<
                                       fx,0,cx,
            0,fy,cy,
            0,0,1);
    int image_index=0;//图像索引
    string dataset = dataset_dir + "/associate.txt";//数据文件
    ifstream fin(dataset);//读取数据集文件
    string rgb_time, rgb_file, depth_time, depth_file;//存放每次读取到的信息
    cv::Mat image_prev, image_this, depth_prev,depth_this;//图片信息
    cv::Mat image_show;//用于展示的图像
    vector<cv::Point2f> points_prev,points_this;//暂存相邻帧角点
    cv::namedWindow("VO",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准

    while (true){
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_file = dataset_dir + "/" + rgb_file;
        depth_file = dataset_dir + "/" + depth_file;
        if (fin.eof())   break;

        image_this = cv::imread(rgb_file);
        image_this.copyTo(image_show);
        depth_this = cv::imread(depth_file);
        cv::cvtColor(image_this, image_this, cv::COLOR_BGR2GRAY);//转化为灰度图
        //判断是不是第一帧
        if(image_prev.empty()){
            double start_frist = cv::getTickCount();
            getPoints_grid(image_this,grid_size,points_num_grid,points_this);
            double time_frsit = (cv::getTickCount() - start_frist) / (double)cv::getTickFrequency();
            cout<<"time for get points in frist image: "<<time_frsit<<endl;
            //亚像素角点精确化
            cv::cornerSubPix(image_this,points_this,cv::Size(10,10),cv::Size(-1,-1),termCriteria);

        }
        else{
            vector<uchar > status_Fast;//光流中记载异常值
            vector<uchar > status_depth;//深度中记载异常值
            vector<float > err_Fast;
            bool Lost = false;//当前帧是否丢失

            double start_flow = cv::getTickCount();
            cv::calcOpticalFlowPyrLK(image_prev,image_this,points_prev,points_this,status_Fast,err_Fast,cv::Size(21,21),3,termCriteria,0,0.001);
            double time_flow = (cv::getTickCount() - start_flow) / cv::getTickFrequency();
            cout << "time for flow : " << time_flow<<endl;

            reducePoints(points_this,status_Fast);
            reducePoints(points_prev,status_Fast);
            rejectWithF(points_prev,points_this,F_THRESHOLD);//去除outlier
            //检测是否跟踪丢失
            if(points_this.size() == 0) {
                cout<<"all points are loss"<<endl;
                return 0;
            }

            //补点操作
            int num_add = MAXCONER - points_this.size();
            if(num_add > 0){
                cv::Mat mask = setmask(image_this,points_this,MIN_DIST);
                double start_add = cv::getTickCount();
                getPoints(image_this,points_this,mask,num_add);
                double time_add = (cv::getTickCount() - start_add) / cv::getTickFrequency();
                cout <<"time for add point: "<< time_add<<endl;
//                cv::imshow("mask",mask);
            }
        }

        //在当前帧上画出角点
        for (int i = 0; i < points_this.size(); ++i) {
            cv::circle(image_show,points_this[i],3,cv::Scalar(0,255,0),-1,8);//画出点
        }
        cv::imshow("VO",image_show);
        //进行帧迭代
        swap(points_this,points_prev);
        swap(image_this,image_prev) ;
        swap(depth_this,depth_prev);
        image_index++;
        cout<<"points size : "<<points_this.size()<<"\nimage index : "<<image_index<<endl;
        cv::waitKey(0);
    }
    return 0;
}
