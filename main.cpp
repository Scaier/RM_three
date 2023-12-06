#include "windmill.hpp"
#include <chrono>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    // // 读取图像
    // Mat src = imread("../image/target.png");
    // if (src.empty())
    // {
    //     cout << "无法读取图像" << endl;
    //     return -1;
    // }

    // // 设置旋转中心和旋转角度
    // Point2f center(src.cols / 2.0, src.rows / 2.0);
    // double angle = 45.0; // 旋转角度，单位为度
    // double scale = 1.0; // 缩放比例

    // // 获取旋转矩阵
    // Mat rot_mat = getRotationMatrix2D(center, angle, scale);

    // // 计算输出图像的尺寸
    // Rect bbox = RotatedRect(Point2f(), src.size(), angle).boundingRect();

    // // 调整旋转矩阵以适应输出图像的尺寸
    // rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    // rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // // 应用旋转矩阵
    // Mat dst;
    // warpAffine(src, dst, rot_mat, bbox.size());

    // // 显示原图和旋转后的图像
    // imshow("原图", src);
    // imshow("旋转后的图像", dst);

    // // 等待按键，然后关闭窗口
    // waitKey(0);
    // return 0;

    // 读取原始图像（全黑背景图像）和要识别的目标图像
    cv::Mat templ = cv::imread("../image/target.png", cv::IMREAD_GRAYSCALE);

    if (templ.empty())
    {
        cout << "无法读取图像" << endl;
        return -1;
    }

    // Mat src = Mat::zeros(720, 1080, CV_8UC1);

    // Mat img = imread("../image/target.png", cv::IMREAD_GRAYSCALE);

    // Point2f center1(img.cols / 2.0, img.rows / 2.0);

    // Mat rot_mat1 = getRotationMatrix2D(center1, 0.0, 1.0);

    // Rect bbox1 = RotatedRect(Point2f(), img.size(), 45.0).boundingRect();

    // rot_mat1.at<double>(0, 2) += bbox1.width / 2.0 - center1.x;
    // rot_mat1.at<double>(1, 2) += bbox1.height / 2.0 - center1.y;

    // warpAffine(img, img, rot_mat1, bbox1.size());

    // for (int i = 0; i < img.cols; i++)
    // {
    //     for (int j = 0; j < img.rows; j++)
    //     {
    //         src.at<uchar>(i + 100, j + 100) = img.at<uchar>(i, j);
    //     }
    // }

    // imshow("img", img);

    // imshow("src", src);

    // waitKey(0);

    // 设置旋转中心和旋转角度
    // Point2f center(templ.cols / 2.0, templ.rows / 2.0);
    // double angle = 5.0;     // 旋转角度，单位为度
    // double scale = 1.0;     // 缩放比例
    // double everytime = 1.0; // 每次旋转角度
    // double best_try = 0.0;  // 最优匹配度
    // Point best_loc;

    // while (best_try < 0.95)
    // {
    //     // 获取旋转矩阵
    //     Mat rot_mat = getRotationMatrix2D(center, angle, scale);

    //     // 计算输出图像的尺寸
    //     Rect bbox = RotatedRect(Point2f(), templ.size(), angle).boundingRect();

    //     // 调整旋转矩阵以适应输出图像的尺寸
    //     rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    //     rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    //     // 应用旋转矩阵
    //     Mat dst;
    //     warpAffine(templ, dst, rot_mat, bbox.size());

    //     // 使用cv::matchTemplate()函数进行模板匹配
    //     cv::Mat result;
    //     cv::matchTemplate(src, dst, result, cv::TM_CCOEFF_NORMED);

    //     // 使用cv::minMaxLoc()函数找到匹配结果中的最大值位置，即目标图像在原始图像中的位置
    //     cv::Point min_loc, max_loc;
    //     double min_val, max_val;
    //     cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    //     cout << max_val << ' ' << angle << ' ' << best_try << endl;

    //     if (max_val > best_try)
    //     {
    //         best_try = max_val;
    //         best_loc = max_loc;
    //     }
    //     angle += everytime;
    //     if (angle >= 360)
    //     {
    //         angle -= 360;
    //     }
    // }

    // // 画矩形框
    // cv::Point pt1(best_loc.x, best_loc.y), pt2(best_loc.x + img.cols, best_loc.y + img.rows);
    // cv::rectangle(src, pt1, pt2, cv::Scalar(255, 0, 0), 2);

    // imshow("windmill", src);

    // waitKey(0);

    Point2f center(templ.cols / 2.0, templ.rows / 2.0);
    double angle1 = 0.0;  // 旋转角度，单位为度
    double scale = 1.0;   // 缩放比例
    double every_try = 10.0; // 每次前进角度
    int max_try = 36;    // 最大尝试次数

    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    cv::Mat src;
    int second = 0;
    double angle = 0.0;
    while (second < 1000)
    {
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        src = wm.getMat((double)t.count() / 1000);

        //=======================================================//

        // 转换为灰度图
        cv::Mat temp;
        cvtColor(src, temp, COLOR_BGR2GRAY);

        double best_try = 0.0; // 最优匹配度
        Point best_loc;
        Mat best_dst;
        int now_try = 0;

        while (now_try < max_try)
        {
            // 获取旋转矩阵
            Mat rot_mat = getRotationMatrix2D(center, angle1, scale);

            // 计算输出图像的尺寸
            Rect bbox = RotatedRect(Point2f(), templ.size(), angle1).boundingRect();

            // 调整旋转矩阵以适应输出图像的尺寸
            rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
            rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

            // 应用旋转矩阵
            Mat dst;
            warpAffine(templ, dst, rot_mat, bbox.size());

            // 使用cv::matchTemplate()函数进行模板匹配
            cv::Mat result;
            cv::matchTemplate(temp, dst, result, cv::TM_CCOEFF_NORMED);

            // 使用cv::minMaxLoc()函数找到匹配结果中的最大值位置，即目标图像在原始图像中的位置
            cv::Point min_loc, max_loc;
            double min_val, max_val;
            cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

            // cout << max_val << ' ' << angle1 << ' ' << best_try << endl;

            if (max_val > best_try)
            {
                best_try = max_val;
                best_loc = max_loc;
                best_dst = dst;
            }

            angle1 += every_try;

            if (angle1 >= 360)
            {
                angle1 -= 360;
            }

            now_try++;

            if (max_val > 0.6)
            {
                max_try = 10;
                every_try = 5.0;
                break;
            }
        }

        cout << now_try << endl;

        // cout << 1 << endl;

        // 画矩形框
        // cv::Point pt1(max_loc.x, max_loc.y), pt2(max_loc.x + templ.cols, max_loc.y + templ.rows);
        cv::Point pt1(best_loc.x, best_loc.y), pt2(best_loc.x + best_dst.cols, best_loc.y + best_dst.rows);
        cv::rectangle(src, pt1, pt2, cv::Scalar(255, 0, 0), 2);

        // Point center(200, 200);
        // circle(src, center, 100, cv::Scalar(255, 0, 0), 10);
        // cv::putText(src, "R", Point(center.x - 5, center.y + 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        imshow("windmill", src);

        //=======================================================//

        second++;
        waitKey(50);
    }
}
