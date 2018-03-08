(define-module (aten)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (gnu packages boost)
  #:use-module (gnu packages maths)
  #:use-module (gnu packages python)
  #:use-module (guix packages)
  #:use-module (guix build-system cmake)
  #:use-module (boost-compute)
  #:use-module (clBLAS))

(define-public aten
  (package
    (name "aten")
    (version "0.4-alpha")
    (source "/home/fis/Workspace/pytorch/pytorch/aten")
    (build-system cmake-build-system)
    (arguments
     `(#:configure-flags
       '("-DNO_CUDA=1")
       #:tests? #f))
    (native-inputs
     `(("python-pyyaml" ,python-pyyaml)
       ("boost" ,boost)
       ("boost-compute" ,boost-compute)
       ("opencl-headers" ,opencl-headers)))
    (inputs `(("openblas" ,openblas)
	      ("clBLAS" ,clBLAS)
	      ("ocl-icd" ,ocl-icd)))
    (home-page "https://github.com/pytorch/pytorch")
    (synopsis "ATen is a simple tensor library")
    (description "ATen is a simple tensor library thats exposes the Tensor
 operations in Torch and PyTorch directly in C++11.")
    (license license:non-copyleft)))
