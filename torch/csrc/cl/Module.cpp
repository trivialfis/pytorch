#include <Python.h>

#include <TH/TH.h>
#include <ATen/ATen.h>

#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/cuda/python_comm.h"
#include "torch/csrc/Exceptions.h"

THCLState *state;

static PyObject * THCLModule_initExtension(PyObject *self)
{
  HANDLE_TH_ERRORS
    state = at::globalContext().lazyInitDevice();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
