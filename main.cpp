#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <ostream>

using namespace std;
using namespace cv;

int GaussianBlur()
{
    cv::Mat src, dst;
    const char *filename = "../picture.jpg";

    cv::imread(filename).copyTo(src);
    if (src.empty())
    {
        throw("Faild open file.");
    }

    int ksize1 = 11;
    int ksize2 = 11;
    double sigma1 = 10.0;
    double sigma2 = 20.0;
    cv::GaussianBlur(src, dst, cv::Size(ksize1, ksize2), sigma1, sigma2);
    //高斯模糊的函数
    //第三，第四，第五参数为高斯模糊的度数

    cv::imshow("src", src);
    cv::imshow("dst", dst);

    return 0;
}

int Red_Wight()
{
    cv::Mat img = cv::imread("../picture.jpg", cv::IMREAD_COLOR);

    int width = img.rows;
    int height = img.cols;

    double _max, _min;
    double r, g, b;
    double h, s, v;
    double c, _h, x;
    double _r, _g, _b;
    double standard1, standard2, standard3;
    double max = 10, min = 0.7;

    cv::Mat hsv = cv::Mat::zeros(width, height, CV_32FC3);
    cv::Mat out = cv::Mat::zeros(width, height, CV_8UC3);

    for (int j = 0; j < width; j++)
    {
        for (int i = 0; i < height; i++)
        {
            // HSV
            r = (float)img.at<cv::Vec3b>(j, i)[2] / 255.0;
            g = (float)img.at<cv::Vec3b>(j, i)[1] / 255.0;
            b = (float)img.at<cv::Vec3b>(j, i)[0] / 255.0;

            _max = fmax(r, fmax(g, b));
            _min = fmin(r, fmin(g, b));

            v = _max;
            if (v > 0)
            {
                s = (_max - _min) / _max;
            }
            else
            {
                s = 0;
            }

            if (_max == r)
            {
                h = 60.0 * (g - b) / (_max - _min);
            }
            else if (_max == g)
            {
                h = 120.0 + 60 * (b - r) / (_max - _min);
            }
            else if (_max == b)
            {
                h = 240.0 + 60 * (r - g) / (_max - _min);
            }
            if (h < 0)
            {
                h = h + 360.0;
            }

            hsv.at<cv::Vec3f>(j, i)[0] = h;
            hsv.at<cv::Vec3f>(j, i)[1] = s;
            hsv.at<cv::Vec3f>(j, i)[2] = v;

            standard1 = hsv.at<cv::Vec3f>(0, 0)[0];
            standard2 = hsv.at<cv::Vec3f>(0, 0)[1];
            standard3 = hsv.at<cv::Vec3f>(0, 0)[2];

            if (h >= min * standard1 && h < max * standard1 && s >= min * standard2 && s <= max * standard2 && v >= min * standard3 && v <= max * standard3)
            {                                                       // Red color detection
                out.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 255, 255); // White background
            }
            else
            {
                out.at<cv::Vec3b>(j, i) = img.at<cv::Vec3b>(j, i); // Black background
            }
        }
    }
    cv::imshow("answer", out);

    return 0;
}

int main(int argc, const char *argv[])
{

    cv::Mat img = imread("../furina.jpeg", IMREAD_COLOR);
    int width = img.rows;
    int height = img.cols;

    // RGB模块
    std::vector<cv::Mat> channels;
    cv::Mat imageBlueChannels;
    cv::Mat imageGreenChannels;
    cv::Mat imageRedChannels;
    // cv::Mat img = cv::imread("../furina.jpeg", cv::IMREAD_COLOR);
    cv::split(img, channels);
    imageBlueChannels = channels.at(0);
    imageGreenChannels = channels.at(1);
    imageRedChannels = channels.at(2);

    cv::imshow("imageBlueChannels", imageBlueChannels);
    cv::imshow("imageGreenChannels", imageGreenChannels);
    cv::imshow("imageRedChannels", imageRedChannels);

    // Mat img = imread("../furina.jpeg", IMREAD_COLOR);
    // Mat img_1 = cv::imread("../furina.jpeg", 0);
    // imshow("ordinary", img);
    // imshow("img_1", img_1);
    // waitKey(0);

    // 灰度图模块

    cv::Mat grey_image = cv::Mat::zeros(width, height, CV_8UC1);

    for (int j = 0; j < width; j++)
    {
        for (int i = 0; i < height; i++)
        {
            grey_image.at<uchar>(j, i) = (int)((float)img.at<cv::Vec3b>(j, i)[0] * 0.0722 +
                                               (float)img.at<cv::Vec3b>(j, i)[1] * 0.7152 +
                                               (float)img.at<cv::Vec3b>(j, i)[2] * 0.2126);
        }
    }

    // HSV图模块
    // 图像读取模块

    double _max, _min;
    double r, g, b;
    double h, s, v;
    double c, _h, x;
    double _r, _g, _b;

    // 建立一个三通道空图
    cv::Mat hsv_image = cv::Mat::zeros(width, height, CV_8UC3);

    for (int j = 0; j < width; j++)
    {
        for (int i = 0; i < height; i++)
        {
            // HSV
            // 提取每个点的RGB数值
            r = (float)img.at<cv::Vec3b>(j, i)[2] / 255;
            g = (float)img.at<cv::Vec3b>(j, i)[1] / 255;
            b = (float)img.at<cv::Vec3b>(j, i)[0] / 255;

            // 提取最大和最小
            _max = fmax(r, fmax(g, b));
            _min = fmin(r, fmin(g, b));

            // 计算h
            if (_max == _min)
            {
                h = 0;
            }
            else if (_min == b)
            {
                h = 60 * (g - r) / (_max - _min) + 60;
            }
            else if (_min == r)
            {
                h = 60 * (b - g) / (_max - _min) + 180;
            }
            else if (_min == g)
            {
                h = 60 * (r - b) / (_max - _min) + 300;
            }
            v = _max;
            s = _max - _min;

            // inverse hue
            h = fmod((h + 180), 360);

            // inverse HSV
            c = s;
            _h = h / 60;
            x = c * (1 - abs(fmod(_h, 2) - 1));

            _r = _g = _b = v - c;

            if (_h < 1)
            {
                _r += c;
                _g += x;
            }
            else if (_h < 2)
            {
                _r += x;
                _g += c;
            }
            else if (_h < 3)
            {
                _g += c;
                _b += x;
            }
            else if (_h < 4)
            {
                _g += x;
                _b += c;
            }
            else if (_h < 5)
            {
                _r += x;
                _b += c;
            }
            else if (_h < 6)
            {
                _r += c;
                _b += x;
            }

            hsv_image.at<cv::Vec3b>(j, i)[0] = (uchar)(_b * 255);
            hsv_image.at<cv::Vec3b>(j, i)[1] = (uchar)(_g * 255);
            hsv_image.at<cv::Vec3b>(j, i)[2] = (uchar)(_r * 255);
        }
    }

    cv::imshow("picture!", img);
    cv::imshow("grey_picture!", grey_image);
    cv::imshow("hsv_picture!", hsv_image);
    GaussianBlur();
    Red_Wight();
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
