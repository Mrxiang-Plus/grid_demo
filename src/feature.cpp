//
// Created by xiang on 2019/10/15.
//
#include "myslam/feature.h"

using namespace std;

//用于sort函数排序
bool  compare(cv::KeyPoint a,cv::KeyPoint b){
    return a.response > b.response;
}

void reducePoints(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reducePoints(std::vector<cv::Point2f> &v, int index ){
    v.erase(v.begin() + index );
}

void minDistance(cv::Mat image, vector<cv::Point2f> &points, int minDistance,int maxCorners){
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

void minDistance(cv::Mat image, vector<cv::KeyPoint> &points, int minDistance,int maxCorners){
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

void getPoints(cv::Mat image,std::vector<cv::Point2f> &points,cv::InputArray mask,int maxConer){
    std::vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
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

void getPoints(cv::Mat image,std::vector<cv::KeyPoint> &points,cv::InputArray mask,int maxConer){
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
    //小于0，无限制，全部加入
    if (maxConer < 0){
        for (auto kp:points_temp)
            points.push_back(kp);
    }

}

void getPoints_grid(cv::Mat image, int grid_size,int point_num_grid, std::vector<cv::Point2f> &points){
    int width = image.cols;
    int height = image.rows;
    int n = width / grid_size; // 一行的grid数；
    int m = height / grid_size; // 一列的grid数
    //每个grid检测
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
            //每块grid检测，并选出最佳point_num_grid个
            mask.colRange(grid_size*i,grid_size*(i+1)).rowRange(grid_size*j,grid_size*(j+1)).setTo(255);
            getPoints(image,points,mask,point_num_grid);
        }
    }
    //grid之外的边缘区域,列边缘，行边缘
    for (int k = 0; k < n; ++k) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*k ,grid_size*(k+1)).rowRange(grid_size*m,height).setTo(255);
        getPoints(image,points,mask,point_num_grid);
    }
    for (int l = 0; l < m; ++l) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*n ,width).rowRange(grid_size*l,grid_size*(l+1)).setTo(255);
        getPoints(image,points,mask,point_num_grid);
    }

    cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
    mask.colRange(grid_size*n ,width).rowRange(grid_size*m,height).setTo(255);
    getPoints(image,points,mask,point_num_grid);
}

void getPoints_grid(cv::Mat image, int grid_size,int point_num_grid, vector<cv::Point2f> &points, int maxConer){
    int width = image.cols;
    int height = image.rows;
    int n = width / grid_size; // 一行的grid数；
    int m = height / grid_size; // 一列的grid数
    vector<cv::KeyPoint> points_temp;
    //每个grid检测
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
            //每块grid检测，并选出最佳point_num_grid个
            mask.colRange(grid_size*i,grid_size*(i+1)).rowRange(grid_size*j,grid_size*(j+1)).setTo(255);
            getPoints(image,points_temp,mask,point_num_grid);
        }
    }
    //grid之外的边缘区域,列边缘，行边缘
    for (int k = 0; k < n; ++k) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*k ,grid_size*(k+1)).rowRange(grid_size*m,height).setTo(255);
        getPoints(image,points_temp,mask,point_num_grid);
    }
    for (int l = 0; l < m; ++l) {
        cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
        mask.colRange(grid_size*n ,width).rowRange(grid_size*l,grid_size*(l+1)).setTo(255);
        getPoints(image,points_temp,mask,point_num_grid);
    }

    cv::Mat mask = cv::Mat::zeros(image.size(),CV_8UC1);
    mask.colRange(grid_size*n ,width).rowRange(grid_size*m,height).setTo(255);
    getPoints(image,points_temp,mask,point_num_grid);

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

void rejectWithF(vector<cv::Point2f> &points1,vector<cv::Point2f> &points2 , double F_THRESHOLD){
    vector<uchar> status;
    if(points2.size() >= 8){
        cv::findFundamentalMat(points1,points2,cv::FM_RANSAC,F_THRESHOLD ,0.99,status);
        reducePoints(points1,status);
        reducePoints(points2,status);

    }
}

cv::Mat setmask(cv::Mat image,vector<cv::Point2f> points, double MIN_DIST){
    cv::Mat mask = cv::Mat(image.size(),CV_8UC1,cv::Scalar(255));
    for (int i = 0; i < points.size(); ++i) {
        if (mask.at<uchar >(points[i]) = 255){
            cv::circle(mask,points[i],MIN_DIST,0,-1);
        }
    }
    return mask;
}

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

void drawTrace(vector<cv::Point2f> &points1,vector<cv::Point2f> &points2,cv::Mat image){
    if(points1.size() != points2.size()) cout<<"wrong point number,can't draw trace"<<endl;
    else
        for (int i = 0; i < points1.size(); ++i) {
            cv::line(image, points1[i], points2[i], cv::Scalar(255, 255, 255), 2, 8);
            cv::circle(image, points2[i], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
        }
}

float getDepth(cv::Mat image_depth,cv::Point2f point, double depth_scale){
    int x = round(point.x);
    int y = round(point.y);

    float d = image_depth.at<float >(point);
    if ( d!=0 )
    {
        return d / depth_scale;
    }
    else
    {
        // check the nearby points
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = image_depth.at<float >((y+dy[i]),(x+dx[i] ));
            if ( d!=0 )
            {
                return d / depth_scale;
            }
        }
    }
    return -1.0;
}