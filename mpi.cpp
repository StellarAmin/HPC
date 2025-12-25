#include "C:/Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/* ================= IMAGE OPERATIONS ================= */

void pointOperations(Mat &image)
{
    Mat gray, dark, contrast, binary, gammaImg;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imwrite("exports/gray_image.png", gray);

    dark = image + Scalar(-50, -50, -50);
    imwrite("exports/dark_image.png", dark);

    image.convertTo(contrast, -1, 1.5, 0);
    imwrite("exports/contrast_image.png", contrast);

    adaptiveThreshold(gray, binary, 230, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2);
    imwrite("exports/binary_image.png", binary);

    imwrite("exports/inverted_image.png", 255 - image);

    // Gamma LUT
    float invGamma = 1.0 / 2.2;
    Mat table(1, 256, CV_8U);
    uchar *p = table.ptr();
    for (int i = 0; i < 256; ++i) p[i] = (uchar)(pow(i / 255.0, invGamma) * 255);
    LUT(image, table, gammaImg);
    imwrite("exports/gamma_image.png", gammaImg);
}

void noise(Mat &image)
{
    Mat snpNoise = image.clone();
    int rows = image.rows, cols = image.cols;
    float amount = 0.004, s_vs_p = 0.5;

    int num_salt = static_cast<int>(ceil(amount * rows * cols * s_vs_p));
    for (int i = 0; i < num_salt; ++i)
        snpNoise.at<Vec3b>(rand() % rows, rand() % cols) = Vec3b(255, 255, 255);

    int num_pepper = static_cast<int>(ceil(amount * rows * cols * (1 - s_vs_p)));
    for (int i = 0; i < num_pepper; ++i)
        snpNoise.at<Vec3b>(rand() % rows, rand() % cols) = Vec3b(0, 0, 0);

    imwrite("exports/snp_noise.png", snpNoise);

    // Gaussian Noise
    Mat gaussNoise(image.size(), image.type());
    RNG rng;
    rng.fill(gaussNoise, RNG::NORMAL, 10, 20);
    imwrite("exports/gaussian_noise.png", image + gaussNoise);

    // Speckle (multiplicative)
    rng.fill(gaussNoise, RNG::NORMAL, 1, 1);
    Mat speckNoise;
    multiply(image, gaussNoise, speckNoise);
    imwrite("exports/speck_noise.png", speckNoise);
}

void filters(Mat &image)
{
    Mat avg, gauss, median, bilateral;
    blur(image, avg, Size(5, 5));
    imwrite("exports/average_blur.png", avg);

    GaussianBlur(image, gauss, Size(5, 5), 0);
    imwrite("exports/gaussian_blur.png", gauss);

    medianBlur(image, median, 5);
    imwrite("exports/median_blur.png", median);

    bilateralFilter(image, bilateral, 9, 75, 75);
    imwrite("exports/bilateral_blur.png", bilateral);
}

void edgeDetection(Mat &image)
{
    Mat sobel, canny, laplacian, sharpened;
    Sobel(image, sobel, CV_8U, 1, 1);
    imwrite("exports/sobel_edge.png", sobel);

    Canny(image, canny, 100, 200);
    imwrite("exports/canny_edge.png", canny);

    Laplacian(image, laplacian, CV_8U);
    imwrite("exports/laplacian_edge.png", laplacian);

    GaussianBlur(image, sharpened, Size(0, 0), 3);
    addWeighted(image, 1.5, sharpened, -0.5, 0, sharpened);
    imwrite("exports/sharpened_image.png", sharpened);
}

void morphing(Mat &image)
{
    Mat eroded, dilated, opened, closed;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));

    erode(image, eroded, element);
    imwrite("exports/eroded_image.png", eroded);

    dilate(image, dilated, element);
    imwrite("exports/dilated_image.png", dilated);

    morphologyEx(image, opened, MORPH_OPEN, element);
    imwrite("exports/opened_image.png", opened);

    morphologyEx(image, closed, MORPH_CLOSE, element);
    imwrite("exports/closed_image.png", closed);

    // Safe reconstruction loop
    Mat marker;
    cvtColor(eroded, marker, COLOR_BGR2GRAY);
    Mat mask;
    cvtColor(image, mask, COLOR_BGR2GRAY);
    Mat kernel = Mat::ones(3, 3, CV_8U);

    while (true)
    {
        Mat expanded;
        dilate(marker, expanded, kernel);
        bitwise_and(expanded, mask, expanded);
        Mat diff;
        compare(marker, expanded, diff, CMP_NE);
        if (countNonZero(diff) == 0) break;
        marker = expanded;
    }
    imwrite("exports/reconstructed_image.png", marker);
}

void geometricTransformations(Mat &image)
{
    Mat rotated, scaled, translated, flipped;
    Point2f center(image.cols / 2.0F, image.rows / 2.0F);
    double angle = 45.0;
    Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(image, rotated, rotMat, image.size());
    imwrite("exports/rotated_image.png", rotated);

    resize(image, scaled, Size(), 1.5, 1.5);
    imwrite("exports/scaled_image.png", scaled);

    Mat transMat = (Mat_<double>(2, 3) << 1, 0, 50, 0, 1, 50);
    warpAffine(image, translated, transMat, image.size());
    imwrite("exports/translated_image.png", translated);

    flip(image, flipped, 1);
    imwrite("exports/flipped_image.png", flipped);

    vector<Point2f> srcPts{{120.23f, 160.75f}, {500.10f, 160.75f}, {380, 400}, {120, 400}};
    vector<Point2f> dstPts{{0, 0}, {640, 0}, {640, 640}, {0, 640}};
    Mat H = getPerspectiveTransform(srcPts, dstPts);
    Mat perspective;
    warpPerspective(image, perspective, H, Size(640, 640), INTER_LINEAR, BORDER_REPLICATE);
    imwrite("exports/perspective_image.png", perspective);
}

void channelOperations(Mat &image)
{
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(hsv, channels);

    // Vectorized HSV adjustments
    Mat hue = channels[0] + 20;
Mat mask = hue >= 180;
subtract(hue, 180, hue, mask);

channels[0] = hue;
channels[1] = min(channels[1] * 2, 255); // saturation
channels[2] = min(channels[2] * 2, 255); // value
        // Hue
    channels[1] = min(channels[1] * 2, 255);        // Saturation
    channels[2] = min(channels[2] * 2, 255);        // Value
    merge(channels, hsv);
    cvtColor(hsv, hsv, COLOR_HSV2BGR);
    imwrite("exports/hsv_adjusted.png", hsv);
}

/* ================= MPI MAIN ================= */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<String> filenames;
    if (rank == 0)
    {
        glob("dataset/*.png", filenames, false);
        if (filenames.empty())
        {
            cerr << "Dataset folder is empty\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    int numImages = filenames.size();
    MPI_Bcast(&numImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double totalStart = MPI_Wtime();

    for (int imgIdx = 0; imgIdx < numImages; imgIdx++)
    {
        Mat image;
        int rows = 0, cols = 0, type = 0;

        if (rank == 0)
        {
            image = imread(filenames[imgIdx]);
            if (!image.data)
            {
                cerr << "Failed to load image\n";
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            rows = image.rows;
            cols = image.cols;
            type = image.type();
        }

        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int baseRows = rows / size;
        int remainder = rows % size;
        int localRows = baseRows + (rank < remainder ? 1 : 0);

        Mat localImage(localRows, cols, type);

        vector<int> sendcounts(size), displs(size);
        if (rank == 0)
        {
            int offset = 0;
            for (int i = 0; i < size; i++)
            {
                int r = baseRows + (i < remainder ? 1 : 0);
                sendcounts[i] = r * cols * CV_ELEM_SIZE(type);
                displs[i] = offset;
                offset += sendcounts[i];
            }
        }

        MPI_Scatterv(rank == 0 ? image.data : nullptr,
                     sendcounts.data(),
                     displs.data(),
                     MPI_UNSIGNED_CHAR,
                     localImage.data,
                     localRows * cols * CV_ELEM_SIZE(type),
                     MPI_UNSIGNED_CHAR,
                     0,
                     MPI_COMM_WORLD);

        // --- Local image processing ---
        pointOperations(localImage);
        noise(localImage);
        filters(localImage);
        edgeDetection(localImage);
        morphing(localImage);
        geometricTransformations(localImage);
        channelOperations(localImage);

        MPI_Gatherv(localImage.data,
                    localRows * cols * CV_ELEM_SIZE(type),
                    MPI_UNSIGNED_CHAR,
                    rank == 0 ? image.data : nullptr,
                    sendcounts.data(),
                    displs.data(),
                    MPI_UNSIGNED_CHAR,
                    0,
                    MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        double totalEnd = MPI_Wtime();
        cout << "Total processing time for " << numImages << " images: "
             << totalEnd - totalStart << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
