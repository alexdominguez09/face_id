import os
import site

# Get the site-packages directory for the current environment
site_packages = site.getsitepackages()[0]
nvidia_base = os.path.join(site_packages, "nvidia")

libraries = [
    "cublas", "cudnn", "cuda_runtime", "cuda_cupti", "cuda_nvcc",
    "cuda_nvrtc", "cufft", "curand", "cusolver", "cusparse", "nccl", "nvjitlink"
]

print("--- NEW CUDA 12 LIBRARY PATHS ---")
found_paths = []
for lib in libraries:
    path = os.path.join(nvidia_base, lib, "lib")
    if os.path.exists(path):
        print(f"{lib}: {path}")
        found_paths.append(path)
    else:
        print(f"{lib}: NOT FOUND")

# This generates the clean string for your LD_LIBRARY_PATH
print("\n--- CLEAN LD_LIBRARY_PATH STRING ---")
print(":".join(found_paths))
