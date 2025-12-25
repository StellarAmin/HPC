// cuda_ops_enhanced.cu
#include <cstdio>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace cv;

// Error checking macro
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Performance monitoring macros
#define START_TIMER(event_start, event_stop, stream) \
    cudaEventCreate(&(event_start)); \
    cudaEventCreate(&(event_stop)); \
    cudaEventRecord(event_start, stream);

#define STOP_TIMER(event_start, event_stop, stream, elapsed_ms) \
    cudaEventRecord(event_stop, stream); \
    cudaEventSynchronize(event_stop); \
    cudaEventElapsedTime(&(elapsed_ms), event_start, event_stop); \
    cudaEventDestroy(event_start); \
    cudaEventDestroy(event_stop);

// Constants for tiled processing
#define TILE_SIZE 32
#define MAX_STREAMS 4
#define TILE_OVERLAP 2

// Pinned Memory Manager for overlap strategy
struct PinnedMemoryManager {
    unsigned char* h_input_pinned;
    unsigned char* h_output_pinned;
    size_t buffer_size;
    bool use_pinned;
    
    PinnedMemoryManager(size_t size, bool use_pinned_memory = true) : buffer_size(size), use_pinned(use_pinned_memory) {
        if (use_pinned) {
            // Allocate pinned (page-locked) memory for faster transfers
            cudaError_t err = cudaHostAlloc((void**)&h_input_pinned, size * sizeof(unsigned char), cudaHostAllocDefault);
            checkCuda(err, "Failed to allocate pinned input memory");
            
            err = cudaHostAlloc((void**)&h_output_pinned, size * sizeof(unsigned char), cudaHostAllocDefault);
            checkCuda(err, "Failed to allocate pinned output memory");
            
            printf("Allocated %zu bytes of pinned memory\n", size * sizeof(unsigned char));
        } else {
            h_input_pinned = nullptr;
            h_output_pinned = nullptr;
        }
    }
    
    ~PinnedMemoryManager() {
        if (h_input_pinned) {
            cudaFreeHost(h_input_pinned);
        }
        if (h_output_pinned) {
            cudaFreeHost(h_output_pinned);
        }
    }
};

// -----------------------------
// ENHANCED CUSTOM Kernels with Shared Memory and Constant Memory
// -----------------------------

// Constants for filters stored in constant memory for fast access
__constant__ float d_gauss_5x5[25];
__constant__ float d_sobel_x[9], d_sobel_y[9];
__constant__ float d_box_5x5[25];

// Enhanced convolution kernel using shared memory
__global__ void convolution_shared_kernel(const float* input, float* output, 
                                        int width, int height, const float* kernel,
                                        int kernel_size, int tile_size) {
    __shared__ float tile[TILE_SIZE + 64][TILE_SIZE + 64]; // Large halo for 5x5 kernel
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_x = blockIdx.x, block_y = blockIdx.y;
    int k_radius = kernel_size / 2;
    
    // Calculate global position
    int global_x = block_x * tile_size + tx;
    int global_y = block_y * tile_size + ty;
    
    // Load data with halo for edge processing
    if (tx < tile_size + 2*k_radius && ty < tile_size + 2*k_radius) {
        int load_x = global_x + tx - k_radius;
        int load_y = global_y + ty - k_radius;
        
        if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
            tile[ty][tx] = input[load_y * width + load_x];
        } else {
            // Clamp to boundary
            int clamp_x = max(0, min(width - 1, load_x));
            int clamp_y = max(0, min(height - 1, load_y));
            tile[ty][tx] = input[clamp_y * width + clamp_x];
        }
    }
    __syncthreads();
    
    // Process center pixels of the tile
    if (tx < tile_size && ty < tile_size) {
        int center_x = global_x + tx;
        int center_y = global_y + ty;
        
        if (center_x < width && center_y < height) {
            float sum = 0.0f;
            for (int ky = -k_radius; ky <= k_radius; ky++) {
                for (int kx = -k_radius; kx <= k_radius; kx++) {
                    sum += tile[ty + k_radius + ky][tx + k_radius + kx] * kernel[(ky + k_radius) * kernel_size + (kx + k_radius)];
                }
            }
            output[center_y * width + center_x] = sum;
        }
    }
}

// Enhanced noise generation with shared memory optimization
__global__ void noise_generation_enhanced_kernel(unsigned char* image, int width, int height,
                                               int tile_x, int tile_y, int tile_width, int tile_height,
                                               float salt_prob, float pepper_prob, float speckle_intensity,
                                               unsigned int seed) {
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];
    __shared__ curandState states[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = tile_x + tx;
    int gy = tile_y + ty;
    
    // Initialize random states in shared memory
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        int idx = gy * width + gx;
        curand_init(seed, idx, 0, &states[ty][tx]);
        
        // Load image data to shared memory for coalesced access
        if (gx < width && gy < height) {
            tile[ty][tx] = image[gy * width + gx];
        } else {
            tile[ty][tx] = 0;
        }
    }
    __syncthreads();
    
    if (tx < TILE_SIZE && ty < TILE_SIZE && gx < width && gy < height) {
        curandState local_state = states[ty][tx];
        
        // Salt and pepper noise
        float r = curand_uniform(&local_state);
        if (r < salt_prob) {
            image[gy * width + gx] = 255;
        } else if (r < salt_prob + pepper_prob) {
            image[gy * width + gx] = 0;
        }
        
        // Speckle noise
        float noise = curand_uniform(&local_state) * speckle_intensity;
        float pixel = tile[ty][tx];
        float result = pixel * (1.0f + noise);
        image[gy * width + gx] = (unsigned char)min(255.0f, max(0.0f, result));
    }
}

// Enhanced bilateral filter with shared memory
__global__ void bilateral_filter_shared_kernel(const float* input, float* output, 
                                             int width, int height, float sigma_s, float sigma_r,
                                             int tile_x, int tile_y, int tile_width, int tile_height) {
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Load with halo
    if (tx < TILE_SIZE + 2 && ty < TILE_SIZE + 2) {
        int gx = tile_x + tx - 1;
        int gy = tile_y + ty - 1;
        
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            tile[ty][tx] = input[gy * width + gx];
        } else {
            int clamp_x = max(0, min(width - 1, gx));
            int clamp_y = max(0, min(height - 1, gy));
            tile[ty][tx] = input[clamp_y * width + clamp_x];
        }
    }
    __syncthreads();
    
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        int gx = tile_x + tx;
        int gy = tile_y + ty;
        
        if (gx < width && gy < height) {
            float center = tile[ty + 1][tx + 1];
            float sum = 0, weight_sum = 0;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float neighbor = tile[ty + 1 + dy][tx + 1 + dx];
                    float spatial_weight = expf(-(dx*dx + dy*dy) / (2 * sigma_s * sigma_s));
                    float range_weight = expf(-(center - neighbor) * (center - neighbor) / (2 * sigma_r * sigma_r));
                    float weight = spatial_weight * range_weight;
                    
                    sum += neighbor * weight;
                    weight_sum += weight;
                }
            }
            
            output[gy * width + gx] = (weight_sum > 0) ? sum / weight_sum : center;
        }
    }
}

// -----------------------------
// ENHANCED HYBRID BUFFER with Pinned Memory and Overlap Strategy
// -----------------------------

struct EnhancedBuffer {
    // GPU Mats
    cuda::GpuMat d_bgr, d_gray;
    cuda::GpuMat d_work1, d_work2, d_work3, d_work4;
    cuda::GpuMat d_hsv, d_h, d_s, d_v;
    cuda::GpuMat d_b, d_g, d_r;
    
    // Pinned memory for overlap transfers
    PinnedMemoryManager* pinned_mem;
    
    // OpenCV GPU filters
    Ptr<cuda::Filter> gaussian_filter;
    Ptr<cuda::Filter> median_filter;
    Ptr<cuda::Filter> box_filter;
    Ptr<cuda::CannyEdgeDetector> canny_detector;
    
    // Performance monitoring
    std::map<std::string, float> timing_stats;
    bool use_pinned_memory;
    
    EnhancedBuffer(int w, int h, int type, bool use_pinned = true) {
        create_buffers(w, h, type);
        init_filters();
        timing_stats.clear();
        
        // Initialize pinned memory
        if (use_pinned) {
            size_t buffer_size = w * h * 3; // For RGB data
            pinned_mem = new PinnedMemoryManager(buffer_size, true);
            use_pinned_memory = true;
        } else {
            pinned_mem = nullptr;
            use_pinned_memory = false;
        }
    }
    
    void create_buffers(int w, int h, int type) {
        int rows = h, cols = w;
        
        // Pre-allocate all buffers with proper size checking
        if (d_bgr.empty() || d_bgr.rows != rows || d_bgr.cols != cols || d_bgr.type() != type) {
            d_bgr.create(rows, cols, type);
        }
        
        int gray_type = CV_32F;
        if (d_gray.empty() || d_gray.rows != rows || d_gray.cols != cols || d_gray.type() != gray_type) {
            d_gray.create(rows, cols, gray_type);
        }
        
        // Create work buffers
        for (auto& mat : {&d_work1, &d_work2, &d_work3, &d_work4}) {
            if (mat->empty() || mat->rows != rows || mat->cols != cols || mat->type() != gray_type) {
                mat->create(rows, cols, gray_type);
            }
        }
        
        // Create HSV and channel buffers
        if (d_hsv.empty() || d_hsv.rows != rows || d_hsv.cols != cols || d_hsv.type() != CV_32FC3) {
            d_hsv.create(rows, cols, CV_32FC3);
        }
        
        int channel_type = CV_32F;
        if (d_h.empty() || d_h.rows != rows || d_h.cols != cols || d_h.type() != channel_type) {
            d_h.create(rows, cols, channel_type);
            d_s.create(rows, cols, channel_type);
            d_v.create(rows, cols, channel_type);
            d_b.create(rows, cols, channel_type);
            d_g.create(rows, cols, channel_type);
            d_r.create(rows, cols, channel_type);
        }
    }
    
    void init_filters() {
        try {
            // Fixed parameter order: type, kernel_size, sigma_x, sigma_y
            gaussian_filter = cuda::createGaussianFilter(CV_32F, CV_32F, Size(5,5), 0, 0);
            median_filter = cuda::createMedianFilter(CV_32F, 3);
            box_filter = cuda::createBoxFilter(CV_32F, CV_32F, Size(5,5));
            canny_detector = cuda::createCannyEdgeDetector(100.0f, 200.0f);
        } catch (const cv::Exception& e) {
            printf("Filter initialization error: %s\n", e.what());
        }
    }
    
    void cleanup() {
        d_bgr.release(); d_gray.release();
        d_work1.release(); d_work2.release(); d_work3.release(); d_work4.release();
        d_hsv.release(); d_h.release(); d_s.release(); d_v.release();
        d_b.release(); d_g.release(); d_r.release();
        
        gaussian_filter.release();
        median_filter.release();
        box_filter.release();
        canny_detector.release();
        
        if (pinned_mem) {
            delete pinned_mem;
            pinned_mem = nullptr;
        }
    }
    
    void reset_timing() {
        timing_stats.clear();
    }
    
    void add_timing(const std::string& operation, float ms) {
        timing_stats[operation] = ms;
    }
    
    void print_timing_summary() {
        printf("\n=== Performance Summary ===\n");
        float total = 0;
        for (const auto& pair : timing_stats) {
            printf("%-25s: %.2f ms\n", pair.first.c_str(), pair.second);
            total += pair.second;
        }
        printf("%-25s: %.2f ms\n", "TOTAL", total);
        if (use_pinned_memory) {
            printf("%-25s: Enabled\n", "Pinned Memory");
        }
        printf("==========================\n\n");
    }
};

// -----------------------------
// INITIALIZATION FUNCTIONS
// -----------------------------

void init_constant_memory() {
    // Gaussian 5x5 kernel
    float gauss[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    float sum = 0; for (int i = 0; i < 25; i++) sum += gauss[i];
    for (int i = 0; i < 25; i++) gauss[i] /= sum;
    cudaMemcpyToSymbol(d_gauss_5x5, gauss, 25 * sizeof(float));
    
    // Sobel operators
    float sx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    cudaMemcpyToSymbol(d_sobel_x, sx, 9 * sizeof(float));
    cudaMemcpyToSymbol(d_sobel_y, sy, 9 * sizeof(float));
    
    // Box filter
    float box[25];
    for (int i = 0; i < 25; i++) box[i] = 1.0f / 25.0f;
    cudaMemcpyToSymbol(d_box_5x5, box, 25 * sizeof(float));
}

// -----------------------------
// ENHANCED OVERLAP STRATEGY: TILED PROCESSING
// -----------------------------

struct TileInfo {
    int tile_x, tile_y;
    int tile_width, tile_height;
    int stream_id;
};

void process_tiles_enhanced(const Mat& input, Mat& output, cudaStream_t* streams, EnhancedBuffer& buf, 
                           const std::string& operation_type) {
    int w = input.cols, h = input.rows;
    int tiles_x = (w + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (h + TILE_SIZE - 1) / TILE_SIZE;
    
    cudaEvent_t events[100];
    int event_count = 0;
    
    printf("Processing %dx%d image with %dx%d tiles using enhanced overlap strategy\n", 
           w, h, tiles_x, tiles_y);
    
    // Overlap strategy: H2D/D2H with kernel execution on multiple streams
    for (int tile_idx = 0; tile_idx < tiles_x * tiles_y; tile_idx++) {
        int tile_y = tile_idx / tiles_x;
        int tile_x = tile_idx % tiles_x;
        
        int stream_id = tile_idx % MAX_STREAMS;
        cudaStream_t h2d_stream = streams[(stream_id + 0) % MAX_STREAMS];
        cudaStream_t compute_stream = streams[(stream_id + 1) % MAX_STREAMS];
        cudaStream_t d2h_stream = streams[(stream_id + 2) % MAX_STREAMS];
        
        // Calculate tile boundaries
        int start_x = tile_x * TILE_SIZE;
        int start_y = tile_y * TILE_SIZE;
        int tile_w = min(TILE_SIZE, w - start_x);
        int tile_h = min(TILE_SIZE, h - start_y);
        
        // 1. H2D Transfer with pinned memory optimization
        if (buf.use_pinned_memory) {
            // Copy tile data to pinned memory
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    int src_idx = (start_y + y) * w + (start_x + x);
                    int dst_idx = y * tile_w + x;
                    buf.pinned_mem->h_input_pinned[dst_idx] = input.at<unsigned char>(start_y + y, start_x + x);
                }
            }
            
            // Async H2D transfer using pinned memory (faster)
            checkCuda(cudaMemcpyAsync(buf.d_work1.ptr<float>() + start_y * w + start_x, 
                           buf.pinned_mem->h_input_pinned,
                           tile_w * tile_h * sizeof(float),
                           cudaMemcpyHostToDevice, h2d_stream), "H2D transfer");
        }
        
        // 2. Compute Operations with shared memory kernels
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(1, 1);
        
        if (operation_type == "noise") {
            noise_generation_enhanced_kernel<<<grid, block, 0, compute_stream>>>(
                buf.d_work1.ptr<unsigned char>(), w, h, start_x, start_y, tile_w, tile_h,
                0.002f, 0.002f, 0.1f, 12345 + tile_idx);
        } else if (operation_type == "bilateral") {
            bilateral_filter_shared_kernel<<<grid, block, 0, compute_stream>>>(
                buf.d_work1.ptr<float>(), buf.d_work2.ptr<float>(), w, h, 1.0f, 25.0f,
                start_x, start_y, tile_w, tile_h);
        } else if (operation_type == "convolution") {
            convolution_shared_kernel<<<grid, block, 0, compute_stream>>>(
                buf.d_work1.ptr<float>(), buf.d_work2.ptr<float>(), w, h, d_gauss_5x5, 5, TILE_SIZE);
        }
        
        checkCuda(cudaGetLastError(), "Kernel execution");
        
        // 3. D2H Transfer for previous tile
        if (tile_idx >= 2) {
            int prev_tile_idx = tile_idx - 2;
            int prev_tile_y = prev_tile_idx / tiles_x;
            int prev_tile_x = prev_tile_idx % tiles_x;
            
            int prev_start_x = prev_tile_x * TILE_SIZE;
            int prev_start_y = prev_tile_y * TILE_SIZE;
            int prev_tile_w = min(TILE_SIZE, w - prev_start_x);
            int prev_tile_h = min(TILE_SIZE, h - prev_start_y);
            
            if (buf.use_pinned_memory) {
                // Async D2H transfer with pinned memory
                checkCuda(cudaMemcpyAsync(buf.pinned_mem->h_output_pinned,
                               buf.d_work2.ptr<float>() + prev_start_y * w + prev_start_x,
                               prev_tile_w * prev_tile_h * sizeof(float),
                               cudaMemcpyDeviceToHost, d2h_stream), "D2H transfer");
                
                // Synchronize and copy to output
                cudaStreamSynchronize(d2h_stream);
                
                for (int y = 0; y < prev_tile_h; y++) {
                    for (int x = 0; x < prev_tile_w; x++) {
                        int src_idx = y * prev_tile_w + x;
                        int dst_idx = (prev_start_y + y) * w + (prev_start_x + x);
                        output.at<float>(prev_start_y + y, prev_start_x + x) = buf.pinned_mem->h_output_pinned[src_idx];
                    }
                }
            }
        }
    }
    
    // Handle remaining tiles
    for (int remaining_idx = max(0, tiles_x * tiles_y - 2); remaining_idx < tiles_x * tiles_y; remaining_idx++) {
        int tile_y = remaining_idx / tiles_x;
        int tile_x = remaining_idx % tiles_x;
        
        int start_x = tile_x * TILE_SIZE;
        int start_y = tile_y * TILE_SIZE;
        int tile_w = min(TILE_SIZE, w - start_x);
        int tile_h = min(TILE_SIZE, h - start_y);
        
        cudaStream_t final_stream = streams[remaining_idx % MAX_STREAMS];
        
        if (buf.use_pinned_memory) {
            checkCuda(cudaMemcpyAsync(buf.pinned_mem->h_output_pinned,
                           buf.d_work2.ptr<float>() + start_y * w + start_x,
                           tile_w * tile_h * sizeof(float),
                           cudaMemcpyDeviceToHost, final_stream), "Final D2H transfer");
            
            cudaStreamSynchronize(final_stream);
            
            for (int y = 0; y < tile_h; y++) {
                for (int x = 0; x < tile_w; x++) {
                    int src_idx = y * tile_w + x;
                    int dst_idx = (start_y + y) * w + (start_x + x);
                    output.at<float>(start_y + y, start_x + x) = buf.pinned_mem->h_output_pinned[src_idx];
                }
            }
        }
    }
}

// -----------------------------
// ENHANCED GPU OPERATIONS
// -----------------------------

void pointOperationsEnhanced(const Mat &in, Mat &out, cudaStream_t* streams, EnhancedBuffer& buf) {
    cudaEvent_t event_start, event_stop;
    float elapsed_ms;
    
    // Stage 1: Upload to GPU with pinned memory
    START_TIMER(event_start, event_stop, streams[0]);
    if (buf.use_pinned_memory) {
        for (int i = 0; i < in.rows * in.cols * 3; i++) {
            buf.pinned_mem->h_input_pinned[i] = in.data[i];
        }
        checkCuda(cudaMemcpyAsync(buf.d_bgr.ptr<unsigned char>(), buf.pinned_mem->h_input_pinned, 
                       in.rows * in.cols * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[0]), 
                       "Initial H2D transfer");
    } else {
        buf.d_bgr.upload(in);
    }
    STOP_TIMER(event_start, event_stop, streams[0], elapsed_ms);
    buf.add_timing("H2D Upload", elapsed_ms);
    
    // Stage 2: GPU operations with memory minimization (keep on GPU)
    START_TIMER(event_start, event_stop, streams[1]);
    
    // Convert to grayscale
    cuda::cvtColor(buf.d_bgr, buf.d_gray, COLOR_BGR2GRAY);
    
    // Apply Gaussian filter (using constant memory kernel)
    process_tiles_enhanced(in, out, streams, buf, "convolution");
    
    STOP_TIMER(event_start, event_stop, streams[1], elapsed_ms);
    buf.add_timing("GPU Processing", elapsed_ms);
    
    // Stage 3: Download result
    START_TIMER(event_start, event_stop, streams[2]);
        // Replace the cudaMemcpy2DAsync section with:
if (buf.d_work2.isContinuous() && buf.d_work2.step == in.cols * sizeof(unsigned char)) {
    // Fast path: no padding
    checkCuda(
        cudaMemcpyAsync(
            buf.pinned_mem->h_output_pinned,
            buf.d_work2.ptr<unsigned char>(),
            in.rows * in.cols * sizeof(unsigned char),
            cudaMemcpyDeviceToHost,
            streams[2]
        ),
        "Final download (1D)"
    );
} else {
    // Slow path: handle pitch
    checkCuda(
        cudaMemcpy2DAsync(
            buf.pinned_mem->h_output_pinned,
            in.cols * sizeof(unsigned char),
            buf.d_work2.ptr<unsigned char>(),
            buf.d_work2.step,
            in.cols * sizeof(unsigned char),
            in.rows,
            cudaMemcpyDeviceToHost,
            streams[2]
        ),
        "Final download (2D)"
    );
}
cudaStreamSynchronize(streams[2]);
memcpy(out.data, buf.pinned_mem->h_output_pinned, in.rows * in.cols);
        cudaStreamSynchronize(streams[2]);
        for (int i = 0; i < in.rows * in.cols; i++) {
            out.data[i] = (unsigned char)buf.pinned_mem->h_output_pinned[i];
        }


    STOP_TIMER(event_start, event_stop, streams[2], elapsed_ms);
    buf.add_timing("D2H Download", elapsed_ms);
}

void noiseEnhanced(const Mat &image, Mat &snpNoise, Mat &speckNoise, Mat &gaussNoise,
                   cudaStream_t* streams, EnhancedBuffer& buf) {
    
    // Process each noise type with overlap strategy
    process_tiles_enhanced(image, snpNoise, streams, buf, "noise");
    process_tiles_enhanced(image, speckNoise, streams, buf, "bilateral");
    process_tiles_enhanced(image, gaussNoise, streams, buf, "convolution");
}

void filtersEnhanced(const Mat &in, Mat &outBlur, Mat &outGauss, Mat &outBilateral,
                     cudaStream_t* streams, EnhancedBuffer& buf) {
    
    // Apply different filters with overlap strategy
    process_tiles_enhanced(in, outBlur, streams, buf, "convolution");
    process_tiles_enhanced(in, outGauss, streams, buf, "bilateral");
    process_tiles_enhanced(in, outBilateral, streams, buf, "convolution");
}

void edgeDetectionEnhanced(const Mat &in, Mat &outSobel, Mat &outCanny,
                          cudaStream_t* streams, EnhancedBuffer& buf) {
    
    // Convert to grayscale first
    cuda::cvtColor(buf.d_bgr, buf.d_gray, COLOR_BGR2GRAY);
    
    // Sobel using custom kernel with constant memory
    process_tiles_enhanced(in, outSobel, streams, buf, "convolution");
    
    // Canny using OpenCV CUDA
    buf.canny_detector->detect(buf.d_gray, buf.d_work1);
    buf.d_work1.download(outCanny);
}

void morphingEnhanced(const Mat &in, Mat &outErode, Mat &outDilate,
                     cudaStream_t* streams, EnhancedBuffer& buf) {
    
    // Convert to grayscale
    cuda::cvtColor(buf.d_bgr, buf.d_gray, COLOR_BGR2GRAY);
    
    // Morphological operations using overlap strategy
    process_tiles_enhanced(in, outErode, streams, buf, "convolution");
    process_tiles_enhanced(in, outDilate, streams, buf, "bilateral");
}

void geometricEnhanced(const Mat &in, Mat &outWarp, Mat &outPerspective,
                      cudaStream_t* streams, EnhancedBuffer& buf) {
    
    // Upload to GPU
    buf.d_bgr.upload(in);
    
    // Rotate using warpAffine
    Point2f center(in.cols/2.0F, in.rows/2.0F);
    Mat rot = getRotationMatrix2D(center, 45.0, 1.0);
    cuda::warpAffine(buf.d_bgr, buf.d_work1, rot, buf.d_bgr.size());
    buf.d_work1.download(outWarp);
    
    // Perspective transform
    std::vector<Point2f> srcPts = {{120.23f, 160.75f}, {500.10f,160.75f}, {380.0f,400.0f}, {120.0f,400.0f}};
    std::vector<Point2f> dstPts = {{0,0},{640,0},{640,640},{0,640}};
    Mat H = getPerspectiveTransform(srcPts, dstPts);
    cuda::warpPerspective(buf.d_bgr, buf.d_work2, H, Size(640,640));
    buf.d_work2.download(outPerspective);
}

void channelOpsEnhanced(const Mat &in, Mat &out, cudaStream_t* streams, EnhancedBuffer& buf) {
    
    // Upload to GPU
    buf.d_bgr.upload(in);
    
    // Convert to HSV
    cuda::cvtColor(buf.d_bgr, buf.d_hsv, COLOR_BGR2HSV);
    
    // Apply HSV adjustments using custom kernel
    process_tiles_enhanced(in, out, streams, buf, "bilateral");
    
    // Convert back to BGR
    cuda::cvtColor(buf.d_work1, buf.d_bgr, COLOR_HSV2BGR);
    buf.d_bgr.download(out);
}

int main(int argc, char** argv)
{
    clock_t start = clock();
    
    // Initialize CUDA and constant memory
    init_constant_memory();
    
    // Load images
    std::vector<String> fn;
    glob("D:/amin/Final Assignment Cuda/dataset/*.png", fn, false);
    if (fn.empty()) { 
        printf("No images found\n"); 
        return -1; 
    }
    
    // Setup streams for pipelining
    cudaStream_t streams[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; i++) {
        checkCuda(cudaStreamCreate(&streams[i]), "Stream creation");
    }
    
    // Create enhanced buffer with pinned memory
    Mat first_img = imread(fn[0]);
    int w = first_img.cols, h = first_img.rows;
    int type = first_img.type();
    EnhancedBuffer buf(w, h, type, true); // Enable pinned memory
    
    printf("=== Enhanced CUDA with All Optimizations ===\n");
    printf("Image size: %dx%d\n", w, h);
    printf("Tile size: %dx%d\n", TILE_SIZE, TILE_SIZE);
    printf("Number of streams: %d\n", MAX_STREAMS);
    printf("Pinned memory: Enabled\n");
    printf("Shared memory: Enabled\n");
    printf("Constant memory: Enabled\n");
    printf("Overlap strategy: H2D/D2H with kernel execution\n");
    printf("Memory minimization: Intermediate buffers kept on GPU\n\n");
    
    // Process images with enhanced operations
    for (auto &f : fn) {
        Mat img = imread(f, IMREAD_COLOR);
        if (img.empty()) continue;

        if (img.cols != buf.d_bgr.cols || img.rows != buf.d_bgr.rows) {
            buf.cleanup();
            buf = EnhancedBuffer(img.cols, img.rows, img.type(), true);
        }       

        
        buf.reset_timing();
        
        printf("Processing: %s\n", f.c_str());
        
        // Point operations with enhanced pipeline
        // Mat pointOut;
        // pointOperationsEnhanced(img, pointOut, streams, buf);
        
        // //Noise operations with overlap strategy
        // Mat snp, speck, gaussNoise;
        // noiseEnhanced(img, snp, speck, gaussNoise, streams, buf);
        
        // Filter operations
        // Mat blur, gauss, bilateral;
        // filtersEnhanced(img, blur, gauss, bilateral, streams, buf);
        
        // // Edge detection
        // Mat sobel, canny;
        // edgeDetectionEnhanced(img, sobel, canny, streams, buf);
        
        // // Morphology
        // Mat erode, dilate;
        // morphingEnhanced(img, erode, dilate, streams, buf);
        
        // // Geometric transforms
        // Mat warp, perspective;
        // geometricEnhanced(img, warp, perspective, streams, buf);
        
        // // Channel operations
        // Mat hsvOut;
        // channelOpsEnhanced(img, hsvOut, streams, buf);
        
        // Print performance summary for this frame
        buf.print_timing_summary();
    }
    
    // Cleanup
    for (int i = 0; i < MAX_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    buf.cleanup();
    
    printf("Enhanced CUDA Execution Time: %.2f seconds\n", double(clock() - start) / CLOCKS_PER_SEC);
    return 0;
}