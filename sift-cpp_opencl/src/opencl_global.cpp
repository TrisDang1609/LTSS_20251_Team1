#include "opencl.hpp"

// Global OpenCL API instance (unified lifecycle management)
// This is the single source of truth for the global opencl_api object
opencl::OPENCL_API opencl_api;
