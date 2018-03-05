(define-module (pytorch)
  #:use-module (guix packages)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (guix build-system python)
  #:use-module (gnu packages base)
  #:use-module (gnu packages cmake)
  #:use-module (gnu packages maths)
  #:use-module (gnu packages ninja)
  #:use-module (gnu packages python)
  #:use-module (clBLAS))

(define-public pytorch
  (package
    (name "pytorch")
    (version "0.4-alpha")
    ;; (source (origin
    ;; 	     (method git-fetch)
    ;; 	     (uri (git-reference
    ;; 		   (url "https://github.com/trivialfis/pytorch")
    ;; 		   (commit "70ba50c3d49c0454210ead1ad931aa39520b10f8")))
    ;; 	     (sha256
    ;; 	      (base32
    ;; 	       "a"))))
    (source "/home/fis/Workspace/pytorch/pytorch")
    (build-system python-build-system)
    (arguments
     `(#:tests? #f
       #:phases (modify-phases %standard-phases
		  (add-before 'configure 'setcuda
		    (lambda _
		      (setenv "NO_CUDA" "1")
		      (setenv "DEBUG" "1"))))
       #:configure-flags
       '("develop" "build")))
    (native-inputs `(("cmake" ,cmake)
		     ("ninja" ,ninja)
		     ("openblas" ,openblas)
		     ("python-pyyaml" ,python-pyyaml)
		     ("which" ,which)))
    (inputs `(("python-numpy" ,python-numpy)
	      ("opencl-headers" ,opencl-headers)
	      ("clBLAS" ,clBLAS)))
    (home-page "http://pytorch.org/")
    (synopsis "Tensors and Dynamic neural networks in Python with strong GPU
 acceleration.")
    (description "
PyTorch is a Python package that provides two high-level features:

@item Tensor computation (like NumPy) with strong GPU acceleration
@item Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy and Cython to
 extend PyTorch when needed.")
    (license license:non-copyleft)))

pytorch
