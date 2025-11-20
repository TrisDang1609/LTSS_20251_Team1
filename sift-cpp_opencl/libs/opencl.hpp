#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300 // target OpenCL 3.0
#include <CL/cl.h>

#include <string>
#include <vector>
#include <map>

namespace opencl
{
  class OPENCL_API
  {
  public:
    /*
    Initialize OpenCL platform, device, context, command queue and build program.
    Parameters:
    - preferredPlatformName: Optional substring of the platform name to select a specific vendor
      (e.g., "NVIDIA" for GPUs like RTX 4060). If empty, the first available platform is used.
    - preferredDeviceType: Desired device type (e.g., CL_DEVICE_TYPE_GPU for RTX 4060).
    - buildOptions: Additional compiler options passed to clBuildProgram (e.g., "-cl-fast-relaxed-math").
    Effects:
    - Queries available platforms and devices.
    - Creates an OpenCL context and command queue for the selected device.
    - Builds an (initially empty) program; kernels can be added later via load_kernel_source().
    Returns:
    - CL_SUCCESS on success, or an OpenCL error code on failure.
    */
    cl_int init(const std::string &preferredPlatformName = "",
                cl_device_type preferredDeviceType = CL_DEVICE_TYPE_GPU,
                const std::string &buildOptions = "");

    /*
    Load and compile OpenCL kernel source from file and attach it to the current program.
    Parameters:
    - sourcePath: Path to a .cl file containing one or more kernel functions.
    - additionalBuildOptions: Extra build options to be appended to those set in init().
    Effects:
    - Reads kernel source code from disk.
    - Creates or replaces the OpenCL program object from this source.
    - Compiles and links the program for the selected device.
    Returns:
    - CL_SUCCESS on success, or an OpenCL error code on failure.
    */
    cl_int load_kernel_source(const std::vector<std::string> &paths,
                              const std::string &additionalBuildOptions);

    /*
    Retrieve a kernel object by name from the currently built program.

    OPTIMIZATION: Kernels are cached! If a kernel with the same name was previously
    requested, the cached kernel is returned instead of creating a new one.
    This is critical for performance when kernels are called repeatedly (e.g., SIFT).

    Parameters:
    - kernelName: Name of the kernel function in the .cl source.
    Effects:
    - If kernel exists in cache: returns cached cl_kernel handle (fast!)
    - If kernel not cached: creates new cl_kernel, adds to cache, returns it
    - All kernels are automatically released when release() is called
    Returns:
    - A valid cl_kernel object on success.
    - nullptr if the kernel cannot be created (check OpenCL error codes via clGetProgramInfo or clGetKernelInfo).

    Example usage pattern in SIFT (20+ blur calls):
        // First call: creates kernel and caches it (~0.5ms)
        cl_kernel k1 = get_kernel("gaussian_blur_vertical");

        // Subsequent calls: returns cached kernel (~0.001ms, 500x faster!)
        cl_kernel k2 = get_kernel("gaussian_blur_vertical");  // k2 == k1 (same object!)
    */
    cl_kernel get_kernel(const std::string &kernelName) const;

    /*
    Create a buffer in the device memory associated with the current context.
    Parameters:
    - sizeInBytes: Size of the buffer in bytes.
    - flags: Memory flags specifying usage (e.g., CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY).
    - hostPtr: Optional pointer to host memory for CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR.
    Effects:
    - Allocates a cl_mem buffer on the device (e.g., RTX 4060 GPU memory).
    Returns:
    - A valid cl_mem object on success, or nullptr on failure.
    */
    cl_mem create_buffer(size_t sizeInBytes,
                         cl_mem_flags flags,
                         void *hostPtr) const;

    /*
    Enqueue a write operation to copy data from host memory to a device buffer.
    Parameters:
    - buffer: Destination buffer on the device created by create_buffer().
    - sizeInBytes: Number of bytes to transfer.
    - hostSrc: Pointer to the source data in host memory.
    - blockingWrite: If true, the call blocks until the data is fully copied to the device.
    Effects:
    - Schedules or performs a memory transfer from CPU RAM to GPU memory (RTX 4060).
    Returns:
    - CL_SUCCESS on success, or an OpenCL error code on failure.
    */
    cl_int write_buffer(cl_mem buffer,
                        size_t sizeInBytes,
                        const void *hostSrc,
                        cl_bool blockingWrite) const;

    /*
    Enqueue a read operation to copy data from a device buffer back to host memory.
    Parameters:
    - buffer: Source buffer on the device.
    - sizeInBytes: Number of bytes to transfer.
    - hostDst: Pointer to the destination memory on the host.
    - blockingRead: If true, the call blocks until the data is fully copied to host memory.
    Effects:
    - Schedules or performs a memory transfer from GPU memory (RTX 4060) back to CPU RAM.
    Returns:
    - CL_SUCCESS on success, or an OpenCL error code on failure.
    */
    cl_int read_buffer(cl_mem buffer,
                       size_t sizeInBytes,
                       void *hostDst,
                       cl_bool blockingRead) const;

    /*
    Copy data from one device buffer to another (GPU-to-GPU transfer).
    Parameters:
    - src_buffer: Source buffer on the device.
    - dst_buffer: Destination buffer on the device.
    - sizeInBytes: Number of bytes to transfer.
    Effects:
    - Performs a memory transfer within GPU memory (no host involvement).
    - Critical for SIFT pyramid building to avoid PCIe bottlenecks.
    Returns:
    - CL_SUCCESS on success, or an OpenCL error code on failure.
    */
    cl_int copy_buffer(cl_mem src_buffer,
                       cl_mem dst_buffer,
                       size_t sizeInBytes) const;

    /*
    Print human-readable information about the selected OpenCL device.
    Effects:
    - Queries and prints properties like device name, vendor, compute units, global memory size,
      and supported OpenCL version. Useful to verify that an RTX 4060 (or other target GPU) is used.
    */
    void print_device_info() const;

    /*
    Release all OpenCL resources owned by this object.
    Effects:
    - Releases program, command queue, context, and device references if they are valid.
    - After this call, the object returns to an uninitialized state and init() must be called again
      before any other methods are used.
    */
    void release();

    /*
    Finish all queued OpenCL operations.
    Blocks until all previously enqueued commands have completed.
    Useful for ensuring GPU memory is freed before next operation.
    */
    inline void finish_queue()
    {
      if (queue)
      {
        clFinish(queue);
      }
    }

    /*
    Release specific OpenCL kernel object.
    Parameters:
    - kernel: The cl_kernel object name to release.
    Effects:
    - Releases the kernel resource if no used anymore.
    */
    void release_kernel(cl_kernel kernel) const;

    /*
    Execute an OpenCL kernel with specified work dimensions.
    Parameters:
    - kernel: The kernel to execute
    - work_dim: Number of dimensions (1, 2, or 3)
    - global_work_size: Array specifying global work size for each dimension
    - local_work_size: Array specifying local work size (can be nullptr for auto)
    Returns:
    - CL_SUCCESS on success, or an OpenCL error code on failure.
    */
    cl_int enqueue_kernel(cl_kernel kernel,
                          cl_uint work_dim,
                          const size_t *global_work_size,
                          const size_t *local_work_size = nullptr) const;

    // public flags to check initialization
    cl_int is_initialized = CL_INVALID_VALUE;

  private:
    cl_int already_initialized = false;
    // OpenCL platform identifier, used to specify the platform to use
    cl_platform_id platform = nullptr;
    // OpenCL device identifier, used to specify the device to use
    cl_device_id device = nullptr;
    // OpenCL context, used to manage OpenCL resources
    cl_context context = nullptr;
    // OpenCL command queue, used to queue commands for execution on the device
    cl_command_queue queue = nullptr;
    // OpenCL program, contains all the kernels (.cl files) used in the application
    cl_program program = nullptr;
    // Vector to track all created kernels for automatic cleanup
    mutable std::vector<cl_kernel> created_kernels;
    // Map for kernel caching: kernel_name -> cl_kernel handle
    // Avoids recreating kernels on every get_kernel() call (critical for SIFT performance!)
    mutable std::map<std::string, cl_kernel> kernel_cache;
    /*
        Example: Targeting an NVIDIA GPU like GPU
        The platform could be NVIDIA's OpenCL platform
        The device would be the specific GPU GPU
        The context would manage resources for the GPU
        The command queue would queue commands for execution on the GPU
        The program would contain kernels optimized for the GPU
        The created_kernels vector would track kernels created for the GPU
    */
  };

}