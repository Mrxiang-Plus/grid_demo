//
// Created by xiang on 2019/10/21.
//

/*
 *vins思想补点方式
 * fast角点
 * 加入最小距离限制
 * 结构优化
 * 加入数据统计
 * 点追踪的次数
 * 丢失的帧数
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
    cv::Mat K = (cv::Mat_<double >(3,3)
            <<fx,0,cx,
            0,fy,cy,
            0,0,1);
    int image_index=0;//图像索引
    string dataset = dataset_dir + "/associate.txt"; // 数据文件
    ifstream fin(dataset);// 读取数据集文件
    string rgb_time, rgb_file, depth_time, depth_file; // 存放每次读取到的信息
    cv::Mat image_prev, image_this, depth_prev,depth_this; //图片信息
    cv::Mat image_show; // 用于展示的图像
    vector<cv::Point2f> points_prev,points_this; // 暂存相邻帧角点
    int points_sum = 0,track_sum = 0,frame_lose = 0; // 追踪到的总点数,所有点追踪的总次数,丢失的帧数
    vector<pair<cv::Point2f,int >> points_cnt;//点和点的追踪次数
    vector<int > track_cnt;//当前点的追踪次数
    cv::namedWindow("VO",cv::WINDOW_AUTOSIZE); // 创建一个显示窗口
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
            //追踪次数初始化
            for (int i = 0; i < points_this.size(); ++i) {
                track_cnt.push_back(1);
            }

            //亚像素角点精确化
            cv::cornerSubPix(image_this,points_this,cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            points_sum += points_this.size();//update
        }
        else{
            vector<uchar > status_Fast;//光流中记载异常值
            vector<uchar > status_depth;//深度中记载异常值
            vector<float > err_Fast;
            bool Lost = false;//当前帧是否丢失

            double start_flow = cv::getTickCount();
            cv::calcOpticalFlowPyrLK(image_prev,image_this,points_prev,points_this,status_Fast,err_Fast,cv::Size(21,21),3,termCriteria,0,0.001);
            double time_flow = (cv::getTickCount() - start_flow) / cv::getTickFrequency();
            cout << "\ntime for flow : " << time_flow<<endl;

            reducePoints(track_cnt,status_Fast);

            reducePoints(points_this,status_Fast);
            reducePoints(points_prev,status_Fast);

            rejectWithF(points_prev,points_this,F_THRESHOLD);//去除outlier
            //检测是否跟踪丢失
            if(points_this.size() < 20) {
                frame_lose ++;
                if (points_this.size() == 0){
                    cout<<"all points are loss"<<endl;
                    return 0;
                }
            }

            //补点操作
            int num_add = MAXCONER - points_this.size();
            int before_add = points_this.size();//补点之前的点数
            if(num_add > 0){
                cv::Mat mask = setmask(image_this,points_this,MIN_DIST);
                double start_add = cv::getTickCount();
                getPoints(image_this,points_this,mask,num_add);
                double time_add = (cv::getTickCount() - start_add) / cv::getTickFrequency();
                cout <<"time for add point: "<< time_add<<endl;
//                cv::imshow("mask",mask);
            }
            //更新追踪次数
            for (int i = 0; i < points_this.size(); ++i) {
                if(track_cnt.size() > i){
                    track_cnt[i] += 1;
                } else track_cnt.push_back(1);
            }

            points_sum += (points_this.size() - before_add);//update

        }
        //更新 points_cnt
        for (int i = 0; i < points_this.size(); ++i) {
            points_cnt.push_back(make_pair(points_this[i],track_cnt[i]));
        }
        //更新总的跟踪次数
        track_sum += points_this.size();
        //在当前帧上画出角点
        for (int i = 0; i < points_this.size(); ++i) {
            double len = std::min(1.0, 1.0 * points_cnt[i].second / WINDOW_SIZE);
            cv::circle(image_show,points_this[i],3,cv::Scalar(255 * (1 - len), 0, 255 * len),-1,8);//画出点
        }
        cv::imshow("VO",image_show);
        //进行帧迭代
        swap(points_this,points_prev);
        swap(image_this,image_prev) ;
        swap(depth_this,depth_prev);
        image_index++;
        cout<<"points size : "<<points_this.size()<<"\nimage index : "<<image_index<<endl;
        cout<<"sum points num : "<<points_sum<<"\nsum track num ： "<<track_sum<<"\n num of lose frame : "<<frame_lose<<endl;
        points_cnt.clear();
        cv::waitKey(0);
    }
    return 0;
}

