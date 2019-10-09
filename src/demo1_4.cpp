//
// Created by xiang on 2019/9/24.
//
///使用特定的图片
///测试不同grid下的计算速度
///带显示模块

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>

using namespace std;

int main(int argc, char ** argv){
    if(argc != 2){
        cout<<"wrong input\nusage: flow path_to_dataset"<<endl;
        return 1 ;
    }

    string dataset_path = argv[1];
    string dataset = dataset_path + "/file.txt";
    ifstream fin(dataset);//读取数据集文件
    //一些必要的数据
    int grid_szie[9] = {10,20,30,40,50,60,70,80,90};
    cv::Mat image ;
    string image_file;

    //遍历图片划分grid并提取角点
    while (true){
        fin >> image_file;
        if (fin.eof())   break;
        image = cv::imread(image_file);

        //划分grid
        int width = image.cols;
        int height = image.rows;
        //9种不同grid
        for (int l = 0; l < 9; ++l) {
            cv::Mat image_copy;
            image.copyTo(image_copy);//复制图片操作
            int grid  = grid_szie[l];
            int n = width / grid; // 一行的grid数；
            int m = height / grid; // 一列的grid数
            int num_points = 0;//总的角点数
            //Fast特征检测器
            vector<cv::KeyPoint> points_fast;
            cv::namedWindow("Fast",cv::WINDOW_NORMAL);
            //cv::resizeWindow("Fast",1080,720);

            cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(40,true,cv::FastFeatureDetector::TYPE_9_16);

            double start_Fast = cv::getTickCount();// 检测开始时
            //每个grid块检测
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    //每块grid检测，并选出最佳5个
                    fastDetector -> detect(image.colRange(grid*i,grid*(i+1)).rowRange(grid*j,grid*(j+1)),points_fast);
                    cv::KeyPointsFilter::retainBest(points_fast,5);
                    //角点坐标有相对与grid调整为相对与完整图片
                    for (int k = 0; k < points_fast.size(); ++k) {
                        points_fast[k].pt += cv::Point2f(grid*i,grid*j);
                    }
                    cv::drawKeypoints(image_copy, points_fast, image_copy, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);//在原图上画出角点
                    num_points += points_fast.size();
                }
            }
            //grid之外的边缘区域,列边缘，行边缘
            fastDetector -> detect(image.colRange(grid*n ,width),points_fast);
            for (int k = 0; k < points_fast.size(); ++k) {
                points_fast[k].pt += cv::Point2f(grid*n,grid*m);
            }
            cv::drawKeypoints(image_copy, points_fast, image_copy, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);

            fastDetector -> detect(image.colRange(0,grid*n).rowRange(grid*m,height),points_fast);
            for (int k = 0; k < points_fast.size(); ++k) {
                points_fast[k].pt += cv::Point2f(grid*n,grid*m);
            }
            cv::drawKeypoints(image_copy, points_fast, image_copy, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);

            //画出grid线条
            for (int i = 1; i <= n; ++i) {
                cv::line(image_copy,cv::Point2f(grid*i,0),cv::Point2f(grid*i,grid*m),cv::Scalar(255,255,255));
            }

            for (int i = 1; i <= m; ++i) {
                cv::line(image_copy,cv::Point2f(0,grid*i),cv::Point2f(grid*n,grid*i),cv::Scalar(255,255,255));
            }

            double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间

            cv::imshow("Fast", image_copy);
            cout << "fast point number: " << num_points << "\ntime for fast: " << time_Fast << " s\n" << endl;

            cv::waitKey(-1);
        }
        cout << "next" <<endl;

    }

    return 0;
}
