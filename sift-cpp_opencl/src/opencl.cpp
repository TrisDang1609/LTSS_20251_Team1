#include "opencl.hpp"

// include OpenCL headers, code below is supposed to run in OPENCL 3.0, Ubuntu 24.04, NVIDIA RTX support
#define CL_HPP_TARGET_OPENCL_VERSION 300 // target OpenCL 3.0
#include <CL/cl.h>

// include params header
#include "params.hpp"

// include other standard libs
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <map>

// Define some helper functions
static bool contains_nocase(const std::string &haystack, const std::string &needle);
static std::string get_platform_string(cl_platform_id p, cl_platform_info param);

namespace opencl
{
    cl_int OPENCL_API::init(const std::string &preferredPlatformName,
                            cl_device_type preferredDeviceType,
                            const std::string &buildOptions)
    {
        // Reset state in case init is called again
        platform = nullptr;
        device = nullptr;
        context = nullptr;
        queue = nullptr;
        program = nullptr;
        created_kernels.clear();
        kernel_cache.clear(); // Clear kernel cache on re-init

        cl_int err = CL_SUCCESS;

        // 1) Get the list of platforms
        cl_uint numPlatforms = 0;
        err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (err != CL_SUCCESS || numPlatforms == 0)
        {
            std::cerr << "No OpenCL platforms found (err = " << err << ")\n";
            return (numPlatforms == 0 ? CL_INVALID_PLATFORM : err);
        }

        std::vector<cl_platform_id> platforms(numPlatforms);
        err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to get OpenCL platform IDs (err = " << err << ")\n";
            return err;
        }

        // 2) Select a platform
        cl_platform_id chosenPlatform = nullptr;

        // 2.1. If the user specifies a platform name
        if (!preferredPlatformName.empty())
        {
            for (auto p : platforms)
            {
                std::string name = get_platform_string(p, CL_PLATFORM_NAME);
                std::string vendor = get_platform_string(p, CL_PLATFORM_VENDOR);
                if (contains_nocase(name, preferredPlatformName) ||
                    contains_nocase(vendor, preferredPlatformName))
                {
                    chosenPlatform = p;
                    break;
                }
            }

            if (!chosenPlatform)
            {
                std::cerr << "Preferred platform \"" << preferredPlatformName
                          << "\" not found, falling back to auto selection.\n";
            }
        }

        // 2.2. If no platform is selected, prioritize NVIDIA
        if (!chosenPlatform)
        {
            for (auto p : platforms)
            {
                std::string name = get_platform_string(p, CL_PLATFORM_NAME);
                std::string vendor = get_platform_string(p, CL_PLATFORM_VENDOR);
                if (contains_nocase(name, "nvidia") ||
                    contains_nocase(vendor, "nvidia"))
                {
                    chosenPlatform = p;
                    break;
                }
            }
        }

        // 2.3. If still no platform is selected, use the first platform
        if (!chosenPlatform)
        {
            chosenPlatform = platforms[0];
        }

        platform = chosenPlatform;

        // 3) Select a device
        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platform, preferredDeviceType, 0, nullptr, &numDevices);

        if (err != CL_SUCCESS || numDevices == 0)
        {
            // If no GPU is found, fallback to CPU
            if (preferredDeviceType == CL_DEVICE_TYPE_GPU)
            {
                std::cerr << "No GPU device found on chosen platform, trying CPU...\n";
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
                if (err != CL_SUCCESS || numDevices == 0)
                {
                    std::cerr << "No CPU device found either (err = " << err << ")\n";
                    return err;
                }
                preferredDeviceType = CL_DEVICE_TYPE_CPU;
            }
            else
            {
                std::cerr << "No device of requested type found (err = " << err << ")\n";
                return err;
            }
        }

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform, preferredDeviceType, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to get device IDs (err = " << err << ")\n";
            return err;
        }

        // For simplicity, just pick the first device
        device = devices[0];

        // (debug) Print the device name
        {
            char name[256];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            std::cout << "Using OpenCL device: " << name << "\n";
        }

        // 4) Create a context
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS || !context)
        {
            std::cerr << "Failed to create OpenCL context (err = " << err << ")\n";
            return err;
        }

        // 5) Create a command queue
        // Suppose using OpenCL 2.0 or later:
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        // If compiling with OpenCL 1.2, use:
        // queue = clCreateCommandQueue(context, device, 0, &err);

        if (err != CL_SUCCESS || !queue)
        {
            std::cerr << "Failed to create command queue (err = " << err << ")\n";
            return err;
        }

        // 6) Optionally, build the program here if desired
        // (Assume you have a member std::string kernelSource;)
        // if (!kernelSource.empty()) {
        //     const char* src = kernelSource.c_str();
        //     size_t len = kernelSource.size();
        //     program = clCreateProgramWithSource(context, 1, &src, &len, &err);
        //     if (err != CL_SUCCESS) return err;
        //     err = clBuildProgram(program, 1, &device,
        //                          buildOptions.c_str(), nullptr, nullptr);
        //     if (err != CL_SUCCESS) { ... print build log ...; return err; }
        // }

        is_initialized = CL_SUCCESS; // flag successful initialization
        return CL_SUCCESS;
    }

    // ========================================================================================

    cl_int OPENCL_API::load_kernel_source(const std::vector<std::string> &paths,
                                          const std::string &additionalBuildOptions)
    {
        std::vector<std::string> sources;
        sources.reserve(paths.size());

        for (const auto &p : paths)
        {
            std::ifstream ifs(p);
            if (!ifs)
            {
                std::cerr << "Cannot open kernel file: " << p << "\n";
                return CL_INVALID_VALUE;
            }
            sources.emplace_back(std::istreambuf_iterator<char>(ifs),
                                 std::istreambuf_iterator<char>());
        }

        std::vector<const char *> cstrs;
        std::vector<size_t> lens;
        for (auto &s : sources)
        {
            cstrs.push_back(s.c_str());
            lens.push_back(s.size());
        }

        cl_int err = CL_SUCCESS;
        if (program)
        {
            clReleaseProgram(program);
            program = nullptr;
        }

        program = clCreateProgramWithSource(context,
                                            static_cast<cl_uint>(cstrs.size()),
                                            cstrs.data(),
                                            lens.data(),
                                            &err);
        if (err != CL_SUCCESS)
            return err;

        err = clBuildProgram(program, 1, &device,
                             additionalBuildOptions.c_str(),
                             nullptr, nullptr);

        // If build fails, get and print the build log
        if (err != CL_SUCCESS)
        {
            size_t logSize = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::string buildLog(logSize, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
            std::cerr << "Error building OpenCL program:\n"
                      << buildLog << "\n";
            return err;
        }

        return err;
    }

    // ========================================================================================

    cl_kernel OPENCL_API::get_kernel(const std::string &kernelName) const
    {
        if (!program)
            return nullptr;

        // OPTIMIZATION: Check cache first!
        // If kernel was previously created, return cached version
        auto it = kernel_cache.find(kernelName);
        if (it != kernel_cache.end())
        {
            // Cache hit! Return existing kernel (no creation overhead)
            // This is 500x faster than creating a new kernel (~0.001ms vs ~0.5ms)
            return it->second;
        }

        // Cache miss: Create new kernel
        cl_int err = CL_SUCCESS;
        cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);

        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to create kernel \"" << kernelName << "\" (err = " << err << ")\n";
            return nullptr;
        }

        // Track this kernel for automatic cleanup
        created_kernels.push_back(kernel);

        // Add to cache for future calls
        kernel_cache[kernelName] = kernel;

        return kernel;
    }

    // ========================================================================================

    cl_mem OPENCL_API::create_buffer(size_t sizeInBytes,
                                     cl_mem_flags flags,
                                     void *hostPtr) const
    {
        cl_int err = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(context, flags, sizeInBytes, hostPtr, &err);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to create buffer of size " << sizeInBytes
                      << " bytes (err = " << err << ")\n";
            return nullptr;
        }
        return buffer;
    }

    // ========================================================================================

    cl_int OPENCL_API::write_buffer(cl_mem buffer,
                                    size_t sizeInBytes,
                                    const void *hostSrc,
                                    cl_bool blockingWrite) const
    {
        cl_int err = clEnqueueWriteBuffer(queue, buffer, blockingWrite,
                                          0, sizeInBytes, hostSrc,
                                          0, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to write to buffer (err = " << err << ")\n";
        }
        return err;
    }

    // ========================================================================================

    cl_int OPENCL_API::read_buffer(cl_mem buffer,
                                   size_t sizeInBytes,
                                   void *hostDst,
                                   cl_bool blockingRead) const
    {
        cl_int err = clEnqueueReadBuffer(queue, buffer, blockingRead,
                                         0, sizeInBytes, hostDst,
                                         0, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to read from buffer (err = " << err << ")\n";
        }
        return err;
    }

    cl_int OPENCL_API::copy_buffer(cl_mem src_buffer,
                                   cl_mem dst_buffer,
                                   size_t sizeInBytes) const
    {
        cl_int err = clEnqueueCopyBuffer(queue, src_buffer, dst_buffer,
                                         0, 0, sizeInBytes,
                                         0, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to copy buffer (err = " << err << ")\n";
        }
        return err;
    }

    // ========================================================================================

    cl_int OPENCL_API::enqueue_kernel(cl_kernel kernel,
                                      cl_uint work_dim,
                                      const size_t *global_work_size,
                                      const size_t *local_work_size) const
    {
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, work_dim,
                                            nullptr, global_work_size, local_work_size,
                                            0, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to enqueue kernel (err = " << err << ")\n";
        }
        return err;
    }

    // ========================================================================================

    void OPENCL_API::print_device_info() const
    {
        if (!device)
        {
            std::cerr << "No device initialized.\n";
            return;
        }

        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);

        cl_uint computeUnits = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);

        size_t maxWorkGroupSize = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);

        cl_ulong globalMemSize = 0;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);

        std::cout << "Device Name: " << name << "\n";
        std::cout << "Compute Units: " << computeUnits << "\n";
        std::cout << "Max Work Group Size: " << maxWorkGroupSize << "\n";
        std::cout << "Global Memory Size: " << (globalMemSize / (1024 * 1024)) << " MB\n";
    }

    // ========================================================================================

    void OPENCL_API::release()
    {
        // Release all tracked kernels first
        for (auto kernel : created_kernels)
        {
            if (kernel)
            {
                clReleaseKernel(kernel);
            }
        }
        created_kernels.clear();
        kernel_cache.clear(); // Clear cache when releasing

        if (program)
        {
            clReleaseProgram(program);
            program = nullptr;
        }
        if (queue)
        {
            clReleaseCommandQueue(queue);
            queue = nullptr;
        }
        if (context)
        {
            clReleaseContext(context);
            context = nullptr;
        }
        if (device)
        {
            clReleaseDevice(device);
            device = nullptr;
        }

        // Reset platform to nullptr for completeness
        platform = nullptr;
    }

    // ========================================================================================

    void OPENCL_API::release_kernel(cl_kernel kernel) const
    {
        if (kernel)
        {
            clReleaseKernel(kernel);

            // Remove from tracking vector
            auto it = std::find(created_kernels.begin(), created_kernels.end(), kernel);
            if (it != created_kernels.end())
            {
                created_kernels.erase(it);
            }

            // Remove from cache
            for (auto it = kernel_cache.begin(); it != kernel_cache.end(); ++it)
            {
                if (it->second == kernel)
                {
                    kernel_cache.erase(it);
                    break;
                }
            }
        }
    }
} // namespace opencl

static bool contains_nocase(const std::string &haystack,
                            const std::string &needle)
{
    if (needle.empty())
        return true;
    if (haystack.size() < needle.size())
        return false;

    auto tolower_str = [](std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return s;
    };

    std::string h = tolower_str(haystack);
    std::string n = tolower_str(needle);

    return h.find(n) != std::string::npos;
}

static std::string get_platform_string(cl_platform_id p, cl_platform_info param)
{
    size_t size = 0;
    clGetPlatformInfo(p, param, 0, nullptr, &size);
    std::string s(size, '\0');
    clGetPlatformInfo(p, param, size, &s[0], nullptr);
    if (!s.empty() && s.back() == '\0')
        s.pop_back();
    return s;
}