(use-modules (gnu packages))
(specifications->manifest
 `(; native inputs
   "cmake"
   "gcc-toolchain"
   "binutils"
   "python-setuptools"
   "ninja"
   "which"
   "python-pyyaml"
   "glibc"
   ;; inputs
   "openblas"
   "python-numpy"
   "clBLAS"
   "opencl-headers"
   ;; for --pure
   "python"
   "bash"
   "coreutils"
   "pkg-config"
   "grep"
   "procps"
   "sed"))
