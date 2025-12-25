#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void pointOperations(Mat image)
{
    Mat image_grey, image_dark, image_contrast, image_binary, image_inverted, image_gamma;
    cvtColor(image, image_grey, COLOR_BGR2GRAY);

    image_dark = image + Scalar(-50, -50, -50);

    image_contrast = image * 1.5;

    adaptiveThreshold(image_grey, image_binary, 230, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2);

    // build a lookup table mapping the pixel values[0, 255] to
    // their adjusted gamma values
    float invGamma = 1.0 / 2.2;
    Mat table(1, 256, CV_8U);
    uchar *p = table.ptr();
    for (int i = 0; i < 256; ++i)
    {
        p[i] = (uchar)(pow(i / 255.0, invGamma) * 255);
    }
    LUT(image, table, image_gamma);
}

void noise(Mat image)
{
    // Salt and Pepper Noise
    int row, col, channels;
    row = image.rows;
    col = image.cols;
    channels = image.channels();
    float s_vs_p = 0.5;
    float amount = 0.004;
    Mat snpNoise = image.clone();
    // Salt mode
    float num_salt = ceil(amount * image.rows * image.cols * s_vs_p);
    std::vector<Point> coords;
    for (int i = 0; i < num_salt; ++i)
    {
        coords.push_back(Point(rand() % col, rand() % row));
    }
    for (const auto &coord : coords)
    {
        snpNoise.at<Vec3b>(coord) = Vec3b(255, 255, 255);
    }

    // Pepper mode
    float num_pepper = ceil(amount * image.rows * image.cols * (1. - s_vs_p));
    coords.clear();
    for (int i = 0; i < num_pepper; ++i)
    {
        coords.push_back(Point(rand() % col, rand() % row));
    }
    for (const auto &coord : coords)
    {
        snpNoise.at<Vec3b>(coord) = Vec3b(0, 0, 0);
    }

    Mat speckNoise;
    Mat gaussNoise = Mat::zeros(image.rows, image.cols, image.type());
    RNG rng;
    rng.fill(gaussNoise, RNG::NORMAL, 10, 20);
    rng.fill(gaussNoise, RNG::NORMAL, 1, 1);
    multiply(image, gaussNoise, speckNoise);
}

void filters(Mat image)
{
    Mat avg, gauss, median, bilateral;
    blur(image, avg, Size(5, 5));

    GaussianBlur(image, gauss, Size(5, 5), 0);

    medianBlur(image, median, 5);

    bilateralFilter(image, bilateral, 9, 75, 75);
}

void edgeDetection(Mat image)
{
    Mat sobel, canny, laplacian, sharpened;
    Sobel(image, sobel, CV_8U, 1, 1);

    Canny(image, canny, 100, 200);

    Laplacian(image, laplacian, CV_8U);

    GaussianBlur(image, sharpened, Size(0, 0), 3);
    addWeighted(image, 1.5, sharpened, -0.5, 0, sharpened);
}

void morphing(Mat image)
{
    Mat eroded, dilated, opened, closed;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(image, eroded, element);

    dilate(image, dilated, element);

    morphologyEx(image, opened, MORPH_OPEN, element);

    morphologyEx(image, closed, MORPH_CLOSE, element);

    Mat marker, mask;
    float radius = 0.5;
    cvtColor(eroded, marker, COLOR_BGR2GRAY);
    cvtColor(image, mask, COLOR_BGR2GRAY);
    Mat kernel = Mat::ones(
        2 * radius + 1,
        2 * radius + 1,
        CV_8U);

    while (true)
    {
        Mat expanded;

        // Dilate marker
        dilate(marker, expanded, kernel);

        // Apply mask constraint
        bitwise_and(expanded, mask, expanded);

        // Termination condition: no change after expansion
        if (countNonZero(marker != expanded) == 0)
        {
            break;
        }

        marker = expanded;
    }
}

void geometricTransformations(Mat image)
{
    Mat rotated, scaled, translated, flipped;
    Point2f center(image.cols / 2.0F, image.rows / 2.0F);
    double angle = 45.0;
    double scale = 1.0;
    Mat rotMat = getRotationMatrix2D(center, angle, scale);
    warpAffine(image, rotated, rotMat, image.size());

    resize(image, scaled, Size(), 1.5, 1.5);

    Mat transMat = (Mat_<double>(2, 3) << 1, 0, 50, 0, 1, 50);
    warpAffine(image, translated, transMat, image.size());

    flip(image, flipped, 1);

    std::vector<Point2f> srcPts = {
        {120.23f, 160.75f},
        {500.10f, 160.75f},
        {380.00f, 400.00f},
        {120.0f, 400.00f}};

    std::vector<Point2f> dstPts = {
        {0.0f, 0.0f},
        {640.0f, 0.0f},
        {640.0f, 640.0f},
        {0.0f, 640.0f}};

    Mat Homography = getPerspectiveTransform(srcPts, dstPts);

    Mat perspective;
    warpPerspective(
        image,
        perspective,
        Homography,
        Size(640, 640),
        INTER_LINEAR, // sub-pixel bilinear interpolation
        BORDER_REPLICATE);
}

void channelOperations(Mat image)
{
    std::vector<Mat> channels;
    Mat zeros = Mat::zeros(image.size(), CV_8UC1);
    Mat blueChannel, greenChannel, redChannel, mergedImage;
    split(image, channels);
    merge(std::vector<Mat>{channels[0], zeros, zeros}, blueChannel);
    merge(std::vector<Mat>{zeros, channels[1], zeros}, greenChannel);
    merge(std::vector<Mat>{zeros, zeros, channels[2]}, redChannel);

    std::vector<Mat> mergedChannels = {channels[0], channels[1], channels[2]};
    merge(mergedChannels, mergedImage);

    Mat hsv;

    for (int i = 0; i < 3; i++)
    {
        cvtColor(image, hsv, COLOR_BGR2HSV);
        split(hsv, channels);
        for (int y = 0; y < hsv.rows; ++y)
        {
            cv::Vec3b *row = hsv.ptr<cv::Vec3b>(y);

            for (int x = 0; x < hsv.cols; ++x)
            {
                int h = row[x][0];
                int s = row[x][1];
                int v = row[x][2];

                if (i == 0)
                {
                    // Hue (wrap, never clamp)
                    h += 20;
                    if (h >= 180)
                        h -= 180;
                    if (h < 0)
                        h += 180;
                }
                else if (i == 1)
                {
                    // Saturation (scale + clamp)
                    s = static_cast<int>(s * 2);
                    s = std::min(255, std::max(0, s));
                }
                else if (i == 2)
                {
                    // Value (scale + clamp)
                    v = static_cast<int>(v * 2);
                    v = std::min(255, std::max(0, v));
                }
                row[x] = cv::Vec3b(
                    static_cast<uchar>(h),
                    static_cast<uchar>(s),
                    static_cast<uchar>(v));
            }
        }

        cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
    }

    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

    for (int y = 0; y < lab.rows; ++y)
    {
        cv::Vec3b *row = lab.ptr<cv::Vec3b>(y);

        for (int x = 0; x < lab.cols; ++x)
        {
            // Normalize L to [0,1]
            float L = row[x][0] / 255.0f;

            // Gamma tone mapping
            L = std::pow(L, 2);

            // Contrast around mid-gray
            L = (L - 0.5f) * 10 + 0.5f;

            // Clamp and rescale
            L = std::min(1.0f, std::max(0.0f, L));
            row[x][0] = static_cast<uchar>(L * 255.0f);
        }
    }

    cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);
}

int main(int argc, char **argv)
{
    clock_t start = clock();

    std::vector<String> fn;
    glob("D:/amin/Final Assignment/dataset/*.png", fn, false);
    std::vector<Mat> images;
    size_t count = fn.size(); // number of png files in images folder
    for (size_t i = 0; i < count; i++)
        images.push_back(imread(fn[i]));

    if (!images[0].data)
    {
        printf("No image data \n");
        return -1;
    }

    for (size_t i = 0; i < count; i++)
    {
        pointOperations(images[i]);
        noise(images[i]);
        filters(images[i]);
        edgeDetection(images[i]);
        morphing(images[i]);
        geometricTransformations(images[i]);
        channelOperations(images[i]);
    }
    printf("Execution Time: %.2f seconds\n", double(clock() - start) / CLOCKS_PER_SEC);
    return 0;
}