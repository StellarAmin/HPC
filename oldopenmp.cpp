// #include <stdio.h>
// #include <time.h>
// #include <omp.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;

// void pointOperations(Mat image)
// {
//     Mat image_grey, image_dark, image_contrast, image_binary, image_inverted, image_gamma;
//     cvtColor(image, image_grey, COLOR_BGR2GRAY);
//     imwrite("exports/gray_image.png", image_grey);

//     image_dark = image + Scalar(-50, -50, -50);
//     imwrite("exports/dark_image.png", image_dark);

//     image_contrast = image * 1.5;
//     imwrite("exports/contrast_image.png", image_contrast);

//     adaptiveThreshold(image_grey, image_binary, 230, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2);
//     imwrite("exports/binary_image.png", image_binary);

//     imwrite("exports/inverted_image.png", 255 - image);

//     // build a lookup table mapping the pixel values[0, 255] to
//     // their adjusted gamma values
//     float invGamma = 1.0 / 2.2;
//     Mat table(1, 256, CV_8U);
//     uchar *p = table.ptr();
//     for (int i = 0; i < 256; ++i)
//     {
//         p[i] = (uchar)(pow(i / 255.0, invGamma) * 255);
//     }
//     LUT(image, table, image_gamma);
//     imwrite("exports/gamma_image.png", image_gamma);
// }

// void noise(Mat image)
// {
//     // Salt and Pepper Noise
//     int row, col, channels;
//     row = image.rows;
//     col = image.cols;
//     channels = image.channels();
//     float s_vs_p = 0.5;
//     float amount = 0.004;
//     Mat snpNoise = image.clone();
//     // Salt mode
//     float num_salt = ceil(amount * image.rows * image.cols * s_vs_p);
//     std::vector<Point> coords;
//     for (int i = 0; i < num_salt; ++i)
//     {
//         coords.push_back(Point(rand() % col, rand() % row));
//     }
//     for (const auto &coord : coords)
//     {
//         snpNoise.at<Vec3b>(coord) = Vec3b(255, 255, 255);
//     }

//     // Pepper mode
//     float num_pepper = ceil(amount * image.rows * image.cols * (1. - s_vs_p));
//     coords.clear();
//     for (int i = 0; i < num_pepper; ++i)
//     {
//         coords.push_back(Point(rand() % col, rand() % row));
//     }
//     for (const auto &coord : coords)
//     {
//         snpNoise.at<Vec3b>(coord) = Vec3b(0, 0, 0);
//     }
//     imwrite("exports/snp_noise.png", snpNoise);

//     Mat speckNoise;
//     Mat gaussNoise = Mat::zeros(image.rows, image.cols, image.type());
//     RNG rng;
//     rng.fill(gaussNoise, RNG::NORMAL, 10, 20);
//     imwrite("exports/gaussian_noise.png", image + gaussNoise);
//     rng.fill(gaussNoise, RNG::NORMAL, 1, 1);
//     multiply(image, gaussNoise, speckNoise);
//     imwrite("exports/speck_noise.png", speckNoise);
// }

// void filters(Mat image)
// {
//     Mat avg, gauss, median, bilateral;
//     blur(image, avg, Size(5, 5));
//     imwrite("exports/average_blur.png", avg);

//     GaussianBlur(image, gauss, Size(5, 5), 0);
//     imwrite("exports/gaussian_blur.png", gauss);

//     medianBlur(image, median, 5);
//     imwrite("exports/median_blur.png", median);

//     bilateralFilter(image, bilateral, 9, 75, 75);
//     imwrite("exports/bilateral_blur.png", bilateral);
// }

// void edgeDetection(Mat image)
// {
//     Mat sobel, canny, laplacian, sharpened;
//     Sobel(image, sobel, CV_8U, 1, 1);
//     imwrite("exports/sobel_edge.png", sobel);

//     Canny(image, canny, 100, 200);
//     imwrite("exports/canny_edge.png", canny);

//     Laplacian(image, laplacian, CV_8U);
//     imwrite("exports/laplacian_edge.png", laplacian);

//     GaussianBlur(image, sharpened, Size(0, 0), 3);
//     addWeighted(image, 1.5, sharpened, -0.5, 0, sharpened);
//     imwrite("exports/sharpened_image.png", sharpened);
// }

// void morphing(Mat image)
// {
//     Mat eroded, dilated, opened, closed;
//     Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//     erode(image, eroded, element);
//     imwrite("exports/eroded_image.png", eroded);

//     dilate(image, dilated, element);
//     imwrite("exports/dilated_image.png", dilated);

//     morphologyEx(image, opened, MORPH_OPEN, element);
//     imwrite("exports/opened_image.png", opened);

//     morphologyEx(image, closed, MORPH_CLOSE, element);
//     imwrite("exports/closed_image.png", closed);

//     Mat marker, mask;
//     float radius = 0.5;
//     cvtColor(eroded, marker, COLOR_BGR2GRAY);
//     cvtColor(image, mask, COLOR_BGR2GRAY);
//     Mat kernel = Mat::ones(
//         2 * radius + 1,
//         2 * radius + 1,
//         CV_8U);

//     while (true)
//     {
//         Mat expanded;

//         // Dilate marker
//         dilate(marker, expanded, kernel);

//         // Apply mask constraint
//         bitwise_and(expanded, mask, expanded);

//         // Termination condition: no change after expansion
//         if (countNonZero(marker != expanded) == 0)
//         {
//             imwrite("exports/reconstructed_image.png", expanded);
//             break;
//         }

//         marker = expanded;
//     }
// }

// void geometricTransformations(Mat image)
// {
//     Mat rotated, scaled, translated, flipped;
//     Point2f center(image.cols / 2.0F, image.rows / 2.0F);
//     double angle = 45.0;
//     double scale = 1.0;
//     Mat rotMat = getRotationMatrix2D(center, angle, scale);
//     warpAffine(image, rotated, rotMat, image.size());
//     imwrite("exports/rotated_image.png", rotated);

//     resize(image, scaled, Size(), 1.5, 1.5);
//     imwrite("exports/scaled_image.png", scaled);

//     Mat transMat = (Mat_<double>(2, 3) << 1, 0, 50, 0, 1, 50);
//     warpAffine(image, translated, transMat, image.size());
//     imwrite("exports/translated_image.png", translated);

//     flip(image, flipped, 1);
//     imwrite("exports/flipped_image.png", flipped);

//     std::vector<Point2f> srcPts = {
//         {120.23f, 160.75f},
//         {500.10f, 160.75f},
//         {380.00f, 400.00f},
//         {120.0f, 400.00f}};

//     std::vector<Point2f> dstPts = {
//         {0.0f, 0.0f},
//         {640.0f, 0.0f},
//         {640.0f, 640.0f},
//         {0.0f, 640.0f}};

//     Mat Homography = getPerspectiveTransform(srcPts, dstPts);

//     Mat perspective;
//     warpPerspective(
//         image,
//         perspective,
//         Homography,
//         Size(640, 640),
//         INTER_LINEAR, // sub-pixel bilinear interpolation
//         BORDER_REPLICATE);
//     imwrite("exports/perspective_image.png", perspective);
// }

// void channelOperations(Mat image)
// {
//     std::vector<Mat> channels;
//     Mat zeros = Mat::zeros(image.size(), CV_8UC1);
//     Mat blueChannel, greenChannel, redChannel, mergedImage;
//     split(image, channels);
//     merge(std::vector<Mat>{channels[0], zeros, zeros}, blueChannel);
//     merge(std::vector<Mat>{zeros, channels[1], zeros}, greenChannel);
//     merge(std::vector<Mat>{zeros, zeros, channels[2]}, redChannel);
//     imwrite("exports/blue_channel.png", blueChannel);
//     imwrite("exports/green_channel.png", greenChannel);
//     imwrite("exports/red_channel.png", redChannel);

//     std::vector<Mat> mergedChannels = {channels[0], channels[1], channels[2]};
//     merge(mergedChannels, mergedImage);
//     imwrite("exports/merged_image.png", mergedImage);

//     Mat hsv;

//     for (int i = 0; i < 3; i++)
//     {
//         cvtColor(image, hsv, COLOR_BGR2HSV);
//         split(hsv, channels);
//         for (int y = 0; y < hsv.rows; ++y)
//         {
//             cv::Vec3b *row = hsv.ptr<cv::Vec3b>(y);

//             for (int x = 0; x < hsv.cols; ++x)
//             {
//                 int h = row[x][0];
//                 int s = row[x][1];
//                 int v = row[x][2];

//                 if (i == 0)
//                 {
//                     // Hue (wrap, never clamp)
//                     h += 20;
//                     if (h >= 180)
//                         h -= 180;
//                     if (h < 0)
//                         h += 180;
//                 }
//                 else if (i == 1)
//                 {
//                     // Saturation (scale + clamp)
//                     s = static_cast<int>(s * 2);
//                     s = std::min(255, std::max(0, s));
//                 }
//                 else if (i == 2)
//                 {
//                     // Value (scale + clamp)
//                     v = static_cast<int>(v * 2);
//                     v = std::min(255, std::max(0, v));
//                 }
//                 row[x] = cv::Vec3b(
//                     static_cast<uchar>(h),
//                     static_cast<uchar>(s),
//                     static_cast<uchar>(v));
//             }
//         }

//         cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
//         if (i == 0)
//             imwrite("exports/hue_image.png", hsv);
//         else if (i == 1)
//             imwrite("exports/saturation_image.png", hsv);
//         else if (i == 2)
//             imwrite("exports/value_image.png", hsv);
//     }

//     cv::Mat lab;
//     cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

//     for (int y = 0; y < lab.rows; ++y)
//     {
//         cv::Vec3b *row = lab.ptr<cv::Vec3b>(y);

//         for (int x = 0; x < lab.cols; ++x)
//         {
//             // Normalize L to [0,1]
//             float L = row[x][0] / 255.0f;

//             // Gamma tone mapping
//             L = std::pow(L, 2);

//             // Contrast around mid-gray
//             L = (L - 0.5f) * 10 + 0.5f;

//             // Clamp and rescale
//             L = std::min(1.0f, std::max(0.0f, L));
//             row[x][0] = static_cast<uchar>(L * 255.0f);
//         }
//     }

//     cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);
//     imwrite("exports/lab_image.png", lab);
// }

// int main(int argc, char **argv)
// {
//     clock_t start = clock();

//     std::vector<String> fn;
//     glob("D:/amin/Final Assignment/dataset/*.png", fn, false);
//     std::vector<Mat> images;
//     size_t count = fn.size(); // number of png files in images folder
//     for (size_t i = 0; i < count; i++)
//         images.push_back(imread(fn[i]));

//     if (!images[0].data)
//     {
//         printf("No image data \n");
//         return -1;
//     }

// #pragma omp parallel for

//     for (int i = 0; i < count; i++)
//     {
//         pointOperations(images[i]);
//         noise(images[i]);
//         filters(images[i]);
//         edgeDetection(images[i]);
//         morphing(images[i]);
//         geometricTransformations(images[i]);
//         channelOperations(images[i]);
//     }

//     printf("Execution Time: %.2f seconds\n", double(clock() - start) / CLOCKS_PER_SEC);
//     return 0;
// }