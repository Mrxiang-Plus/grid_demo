//
// Created by xiang on 2019/9/23.
//
///使用数据集合
///光流追踪对比.特征点选取加入grid

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
using namespace std;


int main(int argc, char** argv){
    if(argc != 2){
        cout<<"wrong input\nusage: flow path_to_dataset"<<endl;
        return 1 ;
    }

    //一些必要的数据
    cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);//创建一个显示窗口

    cv::Mat this_image, prev_image;//图片信息
    cv::Mat fast_image;
    vector<cv::Point2f> points[2];//存储采集到的角点
    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准
    int grid = 30;//grid尺寸
    int npoint = 1; //每个grid中的点数

    string dataset_path = argv[1];
    string dataset = dataset_path + "/file.txt";
    bool isfrist = true;
    int num = 0;
    ifstream fin(dataset);//读取数据集文件

    ///读取数据并进行处理
    while(true){
        string picture_file;//存储当前图片地址
        vector<cv::KeyPoint> points_fast;

        fin >> picture_file;
        if (fin.eof())   break;

        this_image = cv::imread(picture_file);
        this_image.copyTo(fast_image);

        //划分grid
        int width = this_image.cols;
        int height = this_image.rows;
        int n = width / grid; // 一行的grid数
        int m = height / grid; // 一列的grid数

//        cv::cvtColor(this_image, this_image, cv::COLOR_BGR2GRAY);
//        cv::imshow("Fast",this_image);
//        cv::waitKey(0);
        //第一帧提取角点
        if (isfrist){
            ///Fast检测器
            cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(20,true,cv::FastFeatureDetector::TYPE_9_16);
            //每个grid检测
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    //每块grid检测，并选出最佳5个
                    fastDetector -> detect(this_image.colRange(grid*i,grid*(i+1)).rowRange(grid*j,grid*(j+1)),points_fast);
                    cv::KeyPointsFilter::retainBest(points_fast,npoint);
                    //角点坐标有相对与grid调整为相对与完整图片
                    for (int k = 0; k < points_fast.size(); ++k) {
                        points_fast[k].pt += cv::Point2f(grid*i,grid*j);//坐标调整
                        points[1].push_back(points_fast[k].pt);//导入当前帧总体特征点
                    }
                    cv::drawKeypoints(fast_image, points_fast, fast_image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);//在原图上画出角点
                }
            }
            //grid之外的边缘区域,列边缘，行边缘
            fastDetector -> detect(this_image.colRange(grid*n ,width),points_fast);
            for (int k = 0; k < points_fast.size(); ++k) {
                points_fast[k].pt += cv::Point2f(grid*n,grid*m);//坐标调整
                points[1].push_back(points_fast[k].pt);//导入当前帧总体特征点
            }
            cv::drawKeypoints(fast_image, points_fast, fast_image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);

            fastDetector -> detect(this_image.colRange(0,grid*n).rowRange(grid*m,height),points_fast);
            for (int k = 0; k < points_fast.size(); ++k) {
                points_fast[k].pt += cv::Point2f(grid*n,grid*m);//坐标调整
                points[1].push_back(points_fast[k].pt);//导入当前帧总体特征点

            }
            cv::drawKeypoints(fast_image, points_fast, fast_image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);

            //画出grid线条
            for (int i = 1; i <= n; ++i) {
                cv::line(fast_image,cv::Point2f(grid*i,0),cv::Point2f(grid*i,grid*m),cv::Scalar(255,255,255));
            }

            for (int i = 1; i <= m; ++i) {
                cv::line(fast_image,cv::Point2f(0,grid*i),cv::Point2f(grid*n,grid*i),cv::Scalar(255,255,255));
            }


//            cv::cornerSubPix(this_image,points[1],cv::Size(10,10),cv::Size(-1,-1),termCriteria);


//            //在当前帧上画出角点
//            for (int i = 0; i < points[1].size(); ++i) {
//                cv::circle(fast_image,points[1][i],3,cv::Scalar(0,255,0),-1,8);//画出点
//            }

            isfrist = false;
        }
            //后续进行光流追踪
        else{
            vector<uchar > status_Fast;
            vector<float > err_Fast;
            cv::calcOpticalFlowPyrLK(prev_image,this_image,points[0],points[1],status_Fast,err_Fast,cv::Size(10,10),3,termCriteria,0,0.001);

            int i,j;
            for (i=0, j=0; i < points[1].size(); ++i) {
                if (!status_Fast[i]) continue;
                cv::line(fast_image, points[0][i], points[1][i], cv::Scalar(255, 255, 255), 2, 8);
                points[1][j++] = points[1][i];//去除追踪失败的点
                cv::circle(fast_image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
            }
            points[1].resize(j);//调整当前帧的特征点数量
        }

        if(points[1].size() == 0) break;
        cout<<"num of points of Fast : "<<points[1].size()<<endl;
        cv::imshow("Fast",fast_image);

        //进行帧迭代
        swap(points[1],points[0]);
        swap(this_image,prev_image)  ;
        points[1].clear();
        num++;
        cout<<"num of image:"<<num<<endl;
        cv::waitKey(-1);
    }
    return 0;
}