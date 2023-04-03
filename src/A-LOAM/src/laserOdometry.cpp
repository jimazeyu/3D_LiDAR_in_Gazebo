#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include "lidarFactor.hpp"

#define DISTORTION 0

using PointType=pcl::PointXYZI;

int corner_correspondence = 0, plane_correspondence = 0;  //找到多少组对应关系

// constexpr是常量，const代表只读，在C++11中做区分
constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5; //每5帧发布一次点云给后端
bool systemInited = false; //是否收到第一帧数据

// 用于存储上一帧的角点和平面点的kd-tree
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// 用于存储当前帧的角点和平面点
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

// 用于存储上一帧的角点和平面点，不再区分极大和次极大
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// 世界坐标系下的位姿
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// 局部坐标系下的位姿，同时也是优化量
// 和数组共享内存，数组作为ceres优化对象，而Eigen对象则可以方便地进行运算
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// 用于存储当前帧的角点和平面点，因为处理速度和发布速度不一致，所以需要用队列来存储
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;// 互斥锁

// 极大角点回调函数
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}
// 次极大角点回调函数
void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}
// 极大平面点回调函数
void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}
// 次极大平面点回调函数
void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}
// 全部点云回调函数
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

// 假设匀速运动，去除点云畸变
void TransformToStart(PointType const *const pi, PointType *const po)
{
    double s;
    if (DISTORTION)
        // VLP的intensity小鼠部分表示了时间戳，所以可以通过点云的intensity来计算s
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    // q_last_curr就是上一段时间内的旋转量，假设这段时间内也匀速运动，那么就可以通过slerp来进行球面插值
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    // 平移差值直接使用线性插值
    Eigen::Vector3d t_point_last = s * t_last_curr;
    // 更新点云的位置
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;
    // 赋值到输出点云
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

int main(int argc,char **argv)
{
    // 初始化节点
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    // 读取发布频率
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    // 订阅各种点云数据
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    // 发布各种点云数据给后端（不再区分极大极小）
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    // 发布里程计
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布只有前端的里程计
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    nav_msgs::Path laserPath; // 将被发布的路径

    int frameCount = 0;//帧计数
    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        // 如果队列中没有数据，则跳过
        if (cornerSharpBuf.empty() || cornerLessSharpBuf.empty() ||
            surfFlatBuf.empty() || surfLessFlatBuf.empty() ||
            fullPointsBuf.empty())
        {
            rate.sleep();
            continue;
        }
        double timeStamp = fullPointsBuf.front()->header.stamp.toSec();
        // 从队列中取出第一帧点云
        {
            mBuf.lock();// lock start

            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();

            mBuf.unlock();// lock end
        }
        // 系统初始化
        if (!systemInited)
        {
            systemInited = true;
            std::cout << "Initialization finished \n";
        }        
        else
        {
            int cornerPointsSharpNum = cornerPointsSharp->points.size();// 极大角点数量
            int surfPointsFlatNum = surfPointsFlat->points.size();// 极大平面点数量
            // 进行多轮迭代
            for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
            {
                // 重置对应数量
                corner_correspondence = 0;
                plane_correspondence = 0;

                // 使用Huber核函数，对于较大的误差，核函数的值不会随着误差的增大而增大，而是保持一个较小的值，可以剔除外点
                //ceres::LossFunction *loss_function = NULL;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                // 四元数求导非常复杂，使用内部参数
                ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                // 实例化求解问题
                ceres::Problem::Options problem_options;
                ceres::Problem problem(problem_options);
                // 显示添加参数块，尤其在调用内部参数时，需要显示添加
                problem.AddParameterBlock(para_q, 4, q_parameterization);
                problem.AddParameterBlock(para_t, 3);

                // 当前在寻找匹配的点
                pcl::PointXYZI pointSel; 
                // kd-tree查找knn返回数组，包括索引和距离
                std::vector<int> pointSearchInd; 
                std::vector<float> pointSearchSqDis; 

                // 循环极大角点，找对应的边线
                for (int i = 0; i < cornerPointsSharpNum; ++i)
                {
                    TransformToStart(&(cornerPointsSharp->points[i]), &pointSel); // 去畸变
                    kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);// 在上一帧点云中搜索最近邻，返回最近邻的索引和距离 
                    int closestPointInd = -1, minPointInd2 = -1;// 最近邻索引，次近邻索引

                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)// 如果找到的点距离超过阈值，直接结束
                    {
                        closestPointInd = pointSearchInd[0];// 取出最近邻的索引
                        int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);// VLP-16的intensity的整数部分是激光点的线号

                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD; // 寻找次近邻，其实就是设置为允许的最大值，如果次近邻的距离全部大于这个值，那么就寻找失败
                        // 找次近邻
                        for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                        {
                            // 不找同一线的点，要竖着构成直线
                            if (int(laserCloudCornerLast->points[j].intensity) == closestPointScanID)
                                continue;

                            // 线号差超过阈值，无需继续寻找
                            if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            // 计算这个点和当前极大角点的距离
                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                            // 更新次近邻
                            if (pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }

                        // 另一个方向搜索（同理）
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            if (int(laserCloudCornerLast->points[j].intensity) == closestPointScanID)
                                continue;

                            if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    if (minPointInd2 >= 0) // 如果找到了次近邻，添加误差项
                    {
                        // 目前点，上一帧最近邻，上一帧次近邻点
                        Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                    cornerPointsSharp->points[i].y,
                                                    cornerPointsSharp->points[i].z);
                        Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                        laserCloudCornerLast->points[closestPointInd].y,
                                                        laserCloudCornerLast->points[closestPointInd].z);
                        Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                        laserCloudCornerLast->points[minPointInd2].y,
                                                        laserCloudCornerLast->points[minPointInd2].z);

                        double s;
                        if (DISTORTION)
                            s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                        else
                            s = 1.0;
                        // 添加误差项
                        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        // 角对应关系计数
                        corner_correspondence++;
                    }
                }
        
                // 循环极大平面点，寻找平面点到平面的对应关系
                for (int i = 0; i < surfPointsFlatNum; ++i)
                {
                    TransformToStart(&(surfPointsFlat->points[i]), &pointSel); // 去畸变
                    kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); // 在上一帧中找最近邻
                    // 这次找三个点构成平面，先找最近邻l,再找l附近线的最近邻m，再找l同线的最近邻n
                    // 感觉只能找侧面，地面就找不出来
                    int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                    {
                        // 取出最近邻的点号
                        closestPointInd = pointSearchInd[0];
                        // 取出最近邻的线号
                        int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                        // 往scanLine增加的方向搜索
                        for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                        {
                            // 线号差超过阈值，无需继续寻找
                            if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            // 计算这个点和当前极大平面点的距离 
                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // 在同一线上，更新2
                            if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            // 不在同一线上，更新3
                            else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        // 往scanLine减小的方向搜索
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            // 线号差超过阈值，无需继续寻找
                            if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // 在同一线上，更新2
                            if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            // 不在同一线上，更新3
                            else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        // 找到了三个点，构成平面，添加误差项
                        if (minPointInd2 >= 0 && minPointInd3 >= 0)
                        {

                            Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                        surfPointsFlat->points[i].y,
                                                        surfPointsFlat->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                            laserCloudSurfLast->points[closestPointInd].y,
                                                            laserCloudSurfLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                            laserCloudSurfLast->points[minPointInd2].y,
                                                            laserCloudSurfLast->points[minPointInd2].z);
                            Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                            laserCloudSurfLast->points[minPointInd3].y,
                                                            laserCloudSurfLast->points[minPointInd3].z);

                            double s;
                            if (DISTORTION)
                                s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            plane_correspondence++;
                        }
                    }
                }            
                
                // 匹配数量不足
                if ((corner_correspondence + plane_correspondence) < 10)
                {
                    printf("less correspondence! *************************************************\n");
                }

                // 优化
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR; //QR分解
                options.max_num_iterations = 4; //最大迭代次数
                options.minimizer_progress_to_stdout = false; //是否输出到终端
                ceres::Solver::Summary summary; // 优化日志
                ceres::Solve(options, &problem, &summary);

                // 计算全局位姿
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }
        }      
        // 发布位姿
        nav_msgs::Odometry laserOdometry;
        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/laser_odom";
        laserOdometry.header.stamp = ros::Time().fromSec(timeStamp);
        laserOdometry.pose.pose.orientation.x = q_w_curr.x();
        laserOdometry.pose.pose.orientation.y = q_w_curr.y();
        laserOdometry.pose.pose.orientation.z = q_w_curr.z();
        laserOdometry.pose.pose.orientation.w = q_w_curr.w();
        laserOdometry.pose.pose.position.x = t_w_curr.x();
        laserOdometry.pose.pose.position.y = t_w_curr.y();
        laserOdometry.pose.pose.position.z = t_w_curr.z();
        pubLaserOdometry.publish(laserOdometry);
        // 发布路径
        geometry_msgs::PoseStamped laserPose;
        laserPose.header = laserOdometry.header;
        laserPose.pose = laserOdometry.pose.pose;
        laserPath.header.stamp = laserOdometry.header.stamp;
        laserPath.poses.push_back(laserPose);
        laserPath.header.frame_id = "/camera_init";
        pubLaserPath.publish(laserPath);        

        // 更新上一帧数据
        // 千万不能直接赋值，他们的指针指向同一个内存地址，在接收到新的点云后，cornerPointsLessSharp内存会被覆盖，导致laserCloudCornerLast也被覆盖
        //laserCloudCornerLast = cornerPointsLessSharp;
        //laserCloudSurfLast = surfPointsLessFlat;
        // 这里交换了内存，因为cornerPointsLessSharp已经没用了，但是laserCloudCornerLast需要保留
        *laserCloudCornerLast = *cornerPointsLessSharp;
        *laserCloudSurfLast = *surfPointsLessFlat;
        
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        // 计数器到时间发布点云
        if (frameCount % skipFrameNum == 0)
        {
            frameCount = 0;

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeStamp);
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeStamp);
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeStamp);
            laserCloudFullRes3.header.frame_id = "/camera";
            pubLaserCloudFullRes.publish(laserCloudFullRes3);
        }
        // 计数
        frameCount++;
        rate.sleep();
    }
    return 0;
}