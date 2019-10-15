//
// Created by xiang on 2019/9/24.
//
//
// Created by xiang on 2019/9/24.
//
///使用特定的图片
///测试不同grid下的计算速度
///数据统计版

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
    int grid_size[10] = {0,10, 20, 30, 40, 50, 60, 70, 80, 90};
    double times[10] = {0,0,0,0,0,0,0,0,0,0};
    int points_num[10] = {0,0,0,0,0,0,0,0,0,0};
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
        for (int l = 0; l < 10; ++l) {
            cv::Mat image_copy;
            image.copyTo(image_copy);//复制图片操作
            double *time = &times[l];
            int *grid  = &grid_size[l];
            int *number = &points_num[l];
            vector<cv::KeyPoint> points_fast;//存放检测区域的角点

            //Fast特征检测器
            cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(40,true,cv::FastFeatureDetector::TYPE_9_16);

            //无grid时
            if (*grid == 0) {
                double start = cv::getTickCount();// 检测开始时
                fastDetector -> detect(image,points_fast);
                double time_0 = (cv::getTickCount() - start) / (double)cv::getTickFrequency();//检测所花的时间
                *time += time_0;
                *number += points_fast.size();
                continue;
            }
            //有grid时
            int n = width / *grid; // 一行的grid数；
            int m = height / *grid; // 一列的grid数
            double start_Fast = cv::getTickCount();// 检测开始时
            //每个grid块检测
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    //每块grid检测，并选出最佳5个
                    fastDetector -> detect(image.colRange(*grid*i,*grid*(i+1)).rowRange(*grid*j,*grid*(j+1)),points_fast);
                    cv::KeyPointsFilter::retainBest(points_fast,5);
                    *number += points_fast.size();
                }
            }
            //grid之外的边缘区域,列边缘，行边缘
            fastDetector -> detect(image.colRange(*grid*n ,width),points_fast);
            *number += points_fast.size();

            fastDetector -> detect(image.colRange(0,*grid*n).rowRange(*grid*m,height),points_fast);
            *number += points_fast.size();

            double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间
            *time += time_Fast;
        }

    }

    for (int k = 0; k < 10; ++k) {
        cout << "fast point number for grid: " << points_num[k] << "\ntime for fast: " << times[k] << " s\n" << endl;
    }
    return 0;
}

