// cuda_ops.cu
#include <cstdio>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>  // core CUDA types (GpuMat) - optional; some OpenCV builds don't include full CUDA modules
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;

#include <curand_kernel.h>

inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}


// Simple GPU kernel: apply 8-bit LUT (gamma) to 3-channel image
__global__ void gammaLUTKernel(unsigned char* src, unsigned char* dst, const unsigned char* lut, int cols, int rows, int step)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    unsigned char *ps = src + y * step + 3 * x;
    unsigned char *pd = dst + y * step + 3 * x;
    pd[0] = lut[ps[0]];
    pd[1] = lut[ps[1]];
    pd[2] = lut[ps[2]];
}

// HSV adjust kernel (simple per-pixel adjustments - hue, sat, val multipliers)
__global__ void hsvAdjustKernel(unsigned char* src, unsigned char* dst, int cols, int rows, int step, int hue_delta, float sat_mul, float val_mul)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    unsigned char *p = src + y * step + 3 * x;
    unsigned char *q = dst + y * step + 3 * x;
    // src is BGR, convert to HSV (fast approx using OpenCV's convert done on GPU before call)
    // Here we assume input already in HSV (0..179,0..255,0..255)
    int h = p[0] + hue_delta;
    if (h < 0) h += 180;
    if (h >= 180) h -= 180;
    int s = min(255, max(0, int(p[1] * sat_mul)));
    int v = min(255, max(0, int(p[2] * val_mul)));
    q[0] = (unsigned char)h;
    q[1] = (unsigned char)s;
    q[2] = (unsigned char)v;
}

// Helper: run LUT (gamma) on GpuMat using kernel
void applyGammaLUT(const cuda::GpuMat& src, cuda::GpuMat& dst, const Mat &lutHost)
{
    dst.create(src.size(), src.type());
    // copy LUT to device
    cuda::GpuMat d_lut(1, 256, CV_8U);
    d_lut.upload(lutHost);
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1)/block.x, (src.rows + block.y - 1)/block.y);
    gammaLUTKernel<<<grid, block>>>((unsigned char*)src.data, (unsigned char*)dst.data, d_lut.ptr<unsigned char>(), src.cols, src.rows, (int)src.step);
    cudaDeviceSynchronize();
}

void pointOperationsGPU(const Mat &in, Mat &out)
{
    cuda::GpuMat d_in, d_tmp, d_gray;
    d_in.upload(in);

    // Darken: add scalar
    cuda::GpuMat d_dark;
    cv::cuda::add(d_in, Scalar(-50, -50, -50), d_dark);

    // Contrast: scale by 1.5
    cuda::GpuMat d_contrast;
    d_in.convertTo(d_contrast, -1, 1.5, 0);

    // Gray
    cv::cuda::cvtColor(d_in, d_gray, COLOR_BGR2GRAY);

    // adaptiveThreshold: no direct CUDA adaptiveThreshold in many OpenCV versions -> fallback to CPU
    Mat hostGray;
    d_gray.download(hostGray);
    Mat bin;
    adaptiveThreshold(hostGray, bin, 230, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2);
    // upload result if needed
    cuda::GpuMat d_bin;
    d_bin.upload(bin);

    // Gamma LUT: build on host, apply on device via kernel
    float invGamma = 1.0f / 2.2f;
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) lut.at<uchar>(i) = (uchar)(pow(i / 255.0f, invGamma) * 255);
    cuda::GpuMat d_gamma;
    applyGammaLUT(d_in, d_gamma, lut);

    // Compose outputs (example: just download gamma corrected image)
    d_gamma.download(out);
}


// Salt & pepper on CPU then upload to GPU (simpler & fast enough for moderate sizes)
void saltPepperCPU(Mat &image, float amount = 0.004f, float s_vs_p = 0.5f)
{
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);

    Mat out = image.clone();
    int channels = image.channels();

    int num_salt = (int)ceil(amount * image.rows * image.cols * s_vs_p);
    for (int i = 0; i < num_salt; ++i) {
        int x = rand() % image.cols;
        int y = rand() % image.rows;
        if (channels == 1)
            out.at<uchar>(y,x) = 255;
        else
            out.at<Vec3b>(y,x) = Vec3b(255,255,255);
    }

    int num_pepper = (int)ceil(amount * image.rows * image.cols * (1.0f - s_vs_p));
    for (int i = 0; i < num_pepper; ++i) {
        int x = rand() % image.cols;
        int y = rand() % image.rows;
        if (channels == 1)
            out.at<uchar>(y,x) = 0;
        else
            out.at<Vec3b>(y,x) = Vec3b(0,0,0);
    }

    image = out;
}

// ---------------- GPU Noise: Salt & Pepper + Speckle (cuRAND) ----------------

__global__ void salt_pepper_kernel(unsigned char *src, unsigned char *dst,
                                   int width, int height, int channels, int step,
                                   float amount, float s_vs_p, unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int pix = y * width + x;
    int idx = y * step + x * channels;

    curandState state;
    curand_init(seed, (unsigned long long)pix, 0, &state);
    float r = curand_uniform(&state); // (0,1]

    if (r < amount)
    {
        float r2 = curand_uniform(&state);
        unsigned char val = (r2 < s_vs_p) ? 255 : 0;
        for (int c = 0; c < channels; ++c)
            dst[idx + c] = val;
    }
    else
    {
        for (int c = 0; c < channels; ++c)
            dst[idx + c] = src[idx + c];
    }
}

__global__ void speckle_kernel(unsigned char *src, unsigned char *dst,
                               int width, int height, int channels, int step,
                               float mean, float stddev, unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int pix = y * width + x;
    int idx = y * step + x * channels;

    curandState state;
    curand_init(seed, (unsigned long long)pix, 0, &state);
    float n = curand_normal(&state); // mean 0, stddev 1
    float factor = mean + stddev * n;

    for (int c = 0; c < channels; ++c)
    {
        int v = (int)(src[idx + c] * factor + 0.5f);
        dst[idx + c] = (unsigned char)min(255, max(0, v));
    }
}

// Additive Gaussian noise kernel (mean, stddev)
__global__ void gaussian_noise_kernel(unsigned char *src, unsigned char *dst,
                                     int width, int height, int channels, int step,
                                     float mean, float stddev, unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int pix = y * width + x;
    int idx = y * step + x * channels;

    curandState state;
    curand_init(seed, (unsigned long long)pix, 0, &state);
    float n = curand_normal(&state); // mean 0, stddev 1
    float noise = mean + stddev * n;

    for (int c = 0; c < channels; ++c)
    {
        int v = (int)(src[idx + c] + noise + 0.5f);
        dst[idx + c] = (unsigned char)min(255, max(0, v));
    }
}

// Host wrapper: produce salt-and-pepper image, speckle (multiplicative Gaussian) image, and additive Gaussian image
void noiseCUDA(const Mat &image, Mat &snpNoise, Mat &speckNoise, Mat &gaussNoise,
               float amount = 0.004f, float s_vs_p = 0.5f,
               float speck_mean = 1.0f, float speck_std = 0.1f,
               float gauss_mean = 0.0f, float gauss_std = 20.0f)
{
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    cuda::GpuMat d_in;
    d_in.upload(image);

    cuda::GpuMat d_snp(height, width, image.type());
    cuda::GpuMat d_speck(height, width, image.type());
    cuda::GpuMat d_gauss(height, width, image.type());

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    unsigned long long seed = (unsigned long long)clock();

    salt_pepper_kernel<<<grid, block>>>((unsigned char *)d_in.data, (unsigned char *)d_snp.data,
                                        width, height, channels, (int)d_in.step, amount, s_vs_p, seed);

    speckle_kernel<<<grid, block>>>((unsigned char *)d_in.data, (unsigned char *)d_speck.data,
                                    width, height, channels, (int)d_in.step, speck_mean, speck_std, seed + 12345);

    gaussian_noise_kernel<<<grid, block>>>((unsigned char *)d_in.data, (unsigned char *)d_gauss.data,
                                           width, height, channels, (int)d_in.step, gauss_mean, gauss_std, seed + 54321);

    checkCuda(cudaGetLastError(), "noise kernels");

    d_snp.download(snpNoise);
    d_speck.download(speckNoise);
    d_gauss.download(gaussNoise);
}

// Compatibility wrapper that preserves original signature (no outputs) and computes all noise variants
void noise(Mat image)
{
    Mat snp, speck, gauss;
    noiseCUDA(image, snp, speck, gauss);
}

// Filters using CUDA module

void filtersGPU(const Mat &in, Mat &outBlur, Mat &outGauss, Mat &outBilateral)
{
    cuda::GpuMat d_in(in), d_tmp;

    // Convert to grayscale (CV_8UC1) for CUDA filters
    cuda::GpuMat d_gray;
    cuda::cvtColor(d_in, d_gray, COLOR_BGR2GRAY);

    Ptr<cuda::Filter> avg = cuda::createBoxFilter(d_gray.type(), d_gray.type(), Size(5,5));
    avg->apply(d_gray, d_tmp);
    d_tmp.download(outBlur);

    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(d_gray.type(), d_gray.type(), Size(5,5), 0);
    gauss->apply(d_gray, d_tmp);
    d_tmp.download(outGauss);

    // Bilateral: only CV_8UC1 or CV_8UC4
    cuda::GpuMat d_bilateral;
    cuda::bilateralFilter(d_gray, d_bilateral, 9, 75.0, 75.0);
    d_bilateral.download(outBilateral);
}


// Edge detection using CUDA
void edgeDetectionGPU(const Mat &in, Mat &outSobel, Mat &outCanny)
{
    cuda::GpuMat d_in(in), d_gray, d_sobelx, d_sobely, d_canny;
    cuda::cvtColor(d_in, d_gray, COLOR_BGR2GRAY);
    Ptr<cv::cuda::Filter> sobelX =
    cv::cuda::createSobelFilter(CV_8UC1, CV_16S, 1, 0, 3);

    Ptr<cv::cuda::Filter> sobelY =
        cv::cuda::createSobelFilter(CV_8UC1, CV_16S, 0, 1, 3);

    sobelX->apply(d_gray, d_sobelx);
    sobelY->apply(d_gray, d_sobely);

    // Canny (CUDA)
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(100.0, 200.0);
    canny->detect(d_gray, d_canny);
    d_canny.download(outCanny);
}

// Morphology using CUDA
void morphingGPU(const Mat &in, Mat &outErode, Mat &outDilate)
{
    cuda::GpuMat d_in(in), d_out;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, Size(5,5));

    // convert to grayscale (CV_8UC1) for CUDA Morphology
    cuda::GpuMat d_gray;
    cuda::cvtColor(d_in, d_gray, COLOR_BGR2GRAY);

    Ptr<cv::cuda::Filter> erodeFilter =
        cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, d_gray.type(), element);
    erodeFilter->apply(d_gray, d_out);
    d_out.download(outErode);

    Ptr<cv::cuda::Filter> dilateFilter =
        cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, d_gray.type(), element);
    dilateFilter->apply(d_gray, d_out);
    d_out.download(outDilate);
}


// Geometric using CUDA
void geometricGPU(const Mat &in, Mat &outWarp, Mat &outPerspective)
{
    cuda::GpuMat d_in(in), d_out;
    // rotate: warpAffine
    Point2f center(in.cols/2.0F, in.rows/2.0F);
    Mat rot = getRotationMatrix2D(center, 45.0, 1.0);
    cuda::warpAffine(d_in, d_out, rot, d_in.size());
    d_out.download(outWarp);

    // perspective
    std::vector<Point2f> srcPts = {{120.23f, 160.75f}, {500.10f,160.75f}, {380.0f,400.0f}, {120.0f,400.0f}};
    std::vector<Point2f> dstPts = {{0,0},{640,0},{640,640},{0,640}};
    Mat H = getPerspectiveTransform(srcPts, dstPts);
    cuda::warpPerspective(d_in, d_out, H, Size(640,640));
    d_out.download(outPerspective);
}

// Example channel operation: HSV adjust using GPU (convert on GPU then kernel)
void channelOpsGPU(const Mat &in, Mat &out)
{
    cuda::GpuMat d_in(in), d_hsv;
    cuda::cvtColor(d_in, d_hsv, COLOR_BGR2HSV);

    // Launch kernel to adjust hue/sat/val: copy to temp GpuMat that kernel writes to
    cuda::GpuMat d_hsv_out(d_hsv.size(), d_hsv.type());
    dim3 block(16,16);
    dim3 grid((d_hsv.cols + block.x - 1)/block.x, (d_hsv.rows + block.y - 1)/block.y);
    // examples: hue +20, sat *2, val *2
    hsvAdjustKernel<<<grid, block>>>((unsigned char*)d_hsv.data, (unsigned char*)d_hsv_out.data, d_hsv.cols, d_hsv.rows, (int)d_hsv.step, 20, 2.0f, 2.0f);
    cudaDeviceSynchronize();

    cuda::cvtColor(d_hsv_out, d_in, COLOR_HSV2BGR);
    d_in.download(out);
}



int main(int argc, char** argv)
{
    clock_t start = clock();

    // Focused smoke test: run only the CUDA noise routines and write outputs
    std::vector<String> fn;
    glob("D:/amin/Final Assignment Cuda/dataset/*.png", fn, false);
    if (fn.empty()) { printf("No images found\n"); return -1; }


    for (auto &f : fn) {
        Mat img = imread(f, IMREAD_COLOR);
        if (img.empty()) continue;
         // --- Point Operations ---
        Mat pointOut;
        pointOperationsGPU(img, pointOut);

        // --- CUDA Filters ---
        Mat blur, gauss, bilateral;
        filtersGPU(img, blur, gauss, bilateral);

        // --- Edge Detection ---
        Mat sobel, canny;
        edgeDetectionGPU(img, sobel, canny);

        // --- Morphology ---
        Mat erode, dilate;
        morphingGPU(img, erode, dilate);

        // --- Geometric Transforms ---
        Mat warp, perspective;
        geometricGPU(img, warp, perspective);

        // --- Channel / HSV Operations ---
        Mat hsvOut;
        channelOpsGPU(img, hsvOut);

        // --- Noise Operations ---
        Mat snp, speck, gaussNoise;
        noiseCUDA(img, snp, speck, gaussNoise);

        // --- Save outputs (optional) ---
        std::string base = f;
        size_t pos = base.find_last_of('.');
        if (pos == std::string::npos) pos = base.size();

    }

    printf("Execution Time: %.2f seconds\n",
           double(clock() - start) / CLOCKS_PER_SEC);
    return 0;
}