//
// Created by xiang on 2019/10/15.
//

/*
 *采用vins思想补点
 * fast角点
 * 加入最小距离限制
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
const int MIN_DIST = 30;//点间距
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
//对取的点按照最小距离进行筛选
void minDistance(cv::Mat image, vector<cv::Point2f> &points, int minDistance=30,int maxCorners=1000){
    size_t i, j, total = points.size(), ncorners = 0;
    vector<cv::Point2f> corners;
    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( int i = 0; i < total; i++ )
        {
            int y = (int)points[i].y;
            int x = (int)points[i].x;

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int y = (int)points[i].y;
            int x = (int)points[i].x;

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
    points.clear();
    points = corners;
}
//对取的点按照最小距离进行筛选,重载
void minDistance(cv::Mat image, vector<cv::KeyPoint> &points, int minDistance=30,int maxCorners=1000){
    size_t i, j, total = points.size(), ncorners = 0;
    vector<cv::KeyPoint> corners;
    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::KeyPoint> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( int i = 0; i < total; i++ )
        {
            int y = (int)points[i].pt.y;
            int x = (int)points[i].pt.x;

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::KeyPoint> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].pt.x;
                            float dy = y - m[j].pt.y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(points[i]);

                corners.push_back(points[i]);
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int y = (int)points[i].pt.y;
            int x = (int)points[i].pt.x;

            corners.push_back(points[i]);
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
    points.clear();
    points = corners;
}
//对传入的图片提取角点
void getPoints(cv::Mat image,vector<cv::Point2f> &points,cv::InputArray mask=cv::noArray(),int maxConer= -1){
    vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
    //Fast检测器
    cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create(20, true);
    FastDetector -> detect(image, points_Fast,mask);
    //按照响应值降序排序
    sort(points_Fast.begin(),points_Fast.end(),compare);

    /*决定导入点的数量
     * maxConer > 0,按照maxConer导入
     * maxConer = 0,不导入
     * maxConer < 0,无限制
     */
    vector<cv::Point2f> points_temp;
    for (int i = 0; i < points_Fast.size(); ++i) {
        points_temp.push_back(points_Fast[i].pt);
    }

    minDistance(image, points_temp,30,maxConer);
    if (maxConer > 0){
        for (int i = 0; i < maxConer && i< points_temp.size(); ++i) {
            points.push_back(points_temp[i]);
        }
    }
    if (maxConer < 0){
        for (auto kp:points_temp)
            points.push_back(kp);
    }



}
//对传入的图片提取角点,重载，KeyPoint版
void getPoints(cv::Mat image,vector<cv::KeyPoint> &points,cv::InputArray mask=cv::noArray(),int maxConer= -1){
    vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
    //Fast检测器
    cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create(20, true);
    FastDetector -> detect(image, points_Fast,mask);
    //按照响应值降序排序
    sort(points_Fast.begin(),points_Fast.end(),compare);

    /*决定导入点的数量
     * maxConer > 0,按照maxConer导入
     * maxConer = 0,不导入
     * maxConer < 0,无限制
     */
    vector<cv::KeyPoint> points_temp;
    for (int i = 0; i < points_Fast.size(); ++i) {
        points_temp.push_back(points_Fast[i]);
    }

    minDistance(image, points_temp,30,maxConer);

    if (maxConer > 0){
        for (int i = 0; i < maxConer; ++i) {
            points.push_back(points_temp[i]);
        }
    }

    if (maxConer < 0){
        for (auto kp:points_temp)
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
//设置mask区域
cv::Mat setmask(cv::Mat image,vector<cv::Point2f> points){
    cv::Mat mask = cv::Mat(image.size(),CV_8UC1,cv::Scalar(255));
    for (int i = 0; i < points.size(); ++i) {
        if (mask.at<uchar >(points[i]) = 255){
            cv::circle(mask,points[i],MIN_DIST,0,-1);
        }
    }
    return mask;
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
            rejectWithF(points_prev,points_this);//去除outlier

            //补点操作
            int num_add = MAXCONER - points_this.size();
            if(num_add > 0){
                cv::Mat mask = setmask(this_image,points_this);
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

        //进行帧迭代
        swap(points_this,points_prev);
        swap(this_image,prev_image)  ;
        num++;
        cout<<"num of image:"<<num<<endl;
        cv::waitKey(-1);
    }
    return 0;
}
