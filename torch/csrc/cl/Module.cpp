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
  auto m = THPObjectPtr(PyImport_ImportModule("torch.cl"));
  if (!m) throw python_error();

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static struct PyMethodDef _THCLPModule_methods[] =
  {
   {"_cl_init", (PyCFunction)THCLModule_initExtension, METH_NOARGS, NULL},
  };
PyMethodDef* THCLPModule_methods()
{
  return _THCLPModule_methods;
}
