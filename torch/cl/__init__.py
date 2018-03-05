r"""
This package adds support for OpenCL tensor types, that implement the same
function as CPU tensors., but they utilize OpenCL devices for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports OpenCL.

:ref:`OpenCL-semantics has more details about working with OpenCL.
"""

_initialized = False


def is_available():
    pass
