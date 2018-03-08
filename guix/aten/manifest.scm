(use-modules (gnu packages))
(specifications->manifest
 `(; native inputs
   "cmake"
   "gcc-toolchain"
   "binutils"
   "python-setuptools"
   "python-pyyaml"
   "glibc"
   ;; inputs
   "boost"
   "boost-compute"
   "clBLAS"
   "ocl-icd"
   "openblas"
   "opencl-headers"
   ;; for --pure
   "python"
   "bash"
   "coreutils"
   "pkg-config"
   "grep"
   "procps"
   "sed"
   ;; "aten"
   ))
