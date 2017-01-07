/*******************
Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************/

#include <Python.h>
#include <stdio.h>
#include "grid_gpu.cuh"

extern "C" {

typedef struct {
          double x; 
          double y;
} double2;
static PyObject* py_simpleAdd(PyObject* self, PyObject* args);
static PyObject* GPUGrid_convgrid(PyObject* self, PyObject* args);
void initGPUGrid();

/*
 * Test Python extension
 */
static PyObject* py_simpleAdd(PyObject* self, PyObject* args)
{
  double x, y;
  PyArg_ParseTuple(args, "dd", &x, &y);
  return Py_BuildValue("d", x*y);
}
int extractIntList(PyObject *list_in, int** p_array_out, int argnum) {

  //Checking input type
  if (!PyList_Check(list_in)) {
     char errmsg[37];
     sprintf(errmsg,"Argument %d must be a list of int\n", argnum);
     printf("Argument %d must be a list of int\n", argnum);
     fprintf(stderr, "%s", errmsg);fflush(0);
     PyErr_SetString(PyExc_TypeError, errmsg); 
     return -1;
  }
  if (!PyFloat_Check(PyList_GetItem(list_in,0))) {
     char errmsg[37];
     sprintf(errmsg,"Argument %d must be a list of int\n", argnum);
     printf("Argument %d must be a list of int\n", argnum);
     fprintf(stderr, "%s", errmsg);fflush(0);
     PyErr_SetString(PyExc_TypeError, errmsg); 
     return -1;
  }
  int list_size = PyList_Size(list_in);
  PyObject* iter = PyObject_GetIter(list_in);
  *p_array_out = (int*)malloc(sizeof(int)*list_size);
  int* array_out = *p_array_out; 
  int q;
  PyObject* item;
  for (q=0;q<PyList_Size(list_in);q++) {
     item = PyIter_Next(iter);
     array_out[q] = (int)PyInt_AsLong(item);
  }
  return 0;
}
int extractFloatList(PyObject *list_in, double** p_array_out, int argnum) {

  //Checking input type
  if (!PyList_Check(list_in)) {
     char errmsg[37];
     sprintf(errmsg,"Argument %d must be a list of float\n", argnum);
     printf("Argument %d must be a list of float\n", argnum);
     fprintf(stderr, "%s", errmsg);fflush(0);
     PyErr_SetString(PyExc_TypeError, errmsg); 
     return -1;
  }
  if (!PyFloat_Check(PyList_GetItem(list_in,0))) {
     char errmsg[37];
     sprintf(errmsg,"Argument %d must be a list of float\n", argnum);
     printf("Argument %d must be a list of float\n", argnum);
     fprintf(stderr, "%s", errmsg);fflush(0);
     PyErr_SetString(PyExc_TypeError, errmsg); 
     return -1;
  }
  int list_size = PyList_Size(list_in);
  PyObject* iter = PyObject_GetIter(list_in);
  *p_array_out = (double*)malloc(sizeof(double)*list_size);
  double* array_out = *p_array_out; 
  int q;
  PyObject* item;
  for (q=0;q<PyList_Size(list_in);q++) {
     item = PyIter_Next(iter);
     array_out[q] = PyFloat_AsDouble(item);
  }
  return 0;
}
int extractComplexList(PyObject *list_in, double2** p_array_out, int argnum) {

  //Checking input type
  if (!PyList_Check(list_in)) {
     char errmsg[37];
     sprintf(errmsg,"Argument %d must be a list of complex\n", argnum);fflush(0);
     printf("Argument %d must be a list of complex\n", argnum);fflush(0);
     fprintf(stderr, "%s", errmsg);
     PyErr_SetString(PyExc_TypeError, errmsg); 
     return -1;
  }
  if (!PyComplex_Check(PyList_GetItem(list_in,0))) {
     char errmsg[37];
     sprintf(errmsg,"Argument %d must be a list of complex\n", argnum);fflush(0);
     printf("Argument %d must be a list of complex\n", argnum);fflush(0);
     fprintf(stderr, "%s", errmsg);
     PyErr_SetString(PyExc_TypeError, errmsg); 
     return -1;
  }

  int list_size = PyList_Size(list_in);
  PyObject* iter = PyObject_GetIter(list_in);
  *p_array_out = (double2*)malloc(sizeof(double2)*list_size);
  double2* array_out = *p_array_out; 
  int q;
  PyObject* item;
  for (q=0;q<PyList_Size(list_in);q++) {
     item = PyIter_Next(iter);
     array_out[q].x = PyComplex_RealAsDouble(item);
     array_out[q].y = PyComplex_ImagAsDouble(item);
  }
  return 0;
}
void makeComplexList(double2* array_in, int list_size, PyObject *list_out) {
  if (!PyList_Check(list_out)) fprintf(stderr,"makeComplexList must take "
                                              "a Python list\n");
  int q;
  for (q=0;q<list_size;q++) {
     PyList_Append(list_out, PyComplex_FromDoubles(array_in[q].x, array_in[q].y)); 
  }
}
static PyObject* GPUGrid_convgrid(PyObject* self, PyObject* args)
{
  int npts, img_size, Qpx, gcf_dim, q;
  PyObject *in, *in_vals, *in_gcfinx, *gcf, *out;
  if(!PyArg_ParseTuple(args, "OiOOiOii", &in, &npts, &in_vals, &in_gcfinx, &img_size, 
                                    &gcf, &Qpx, &gcf_dim)) {
    PyErr_SetString(PyExc_TypeError, "Incorrect number or type of arguments to convgrid.\n\n"
        "Usage: convgrid(in, npts, in_vals, in_gcfinx, img_size, gcf, Qpx, gcf_dim)\n"
        "    in: list of float\n"
        "    npts: integer\n"
        "    in_vals: list of complex\n"
        "    in_gcfinx: list of complex\n"
        "    img_size: integer\n"
        "    gcf: list of complex\n"
        "    Qpx: integer\n"
        "    gcf_dim: integer\n"
    );
    printf("Incorrect number or type of arguments to convgrid.\n");
  }

  double2 *gcf_c, *in_vals_c, *out_c;
  double *in_c;
  int* in_gcfinx_c;
  if (0 != extractIntList(in_gcfinx, &in_gcfinx_c, 4)) return Py_BuildValue("");
  if (0 != extractComplexList(in_vals, &in_vals_c, 3)) return Py_BuildValue("");
  if (0 != extractFloatList(in, &in_c, 1)) return Py_BuildValue("");
  if (0 != extractComplexList(gcf, &gcf_c, 6)) return Py_BuildValue("");
  out_c = (double2*)malloc(sizeof(double2)*npts);
#if 1
  gridGPU(out_c, (double2*)in_c, (double2*)in_vals_c, in_gcfinx_c, npts, img_size, gcf_c, gcf_dim); 
  printf("Done with gridGPU\n"); fflush(0);
#else
  for (q=0;q<npts;q++) {
     out_c[q].x = 2.0*in_c[2*q];
     out_c[q].y = 2.0*in_c[2*q+1];
  }
#endif

  out = PyList_New(0);
  makeComplexList(out_c, img_size*img_size, out);
  printf("len(out) = %lu\n", PyList_Size(out));

  return Py_BuildValue("O", out);
}

/*
 * Bind Python function names to our C functions
 */
static PyMethodDef GPUGrid_methods[] = {
  {"simpleAdd", py_simpleAdd, METH_VARARGS, "Adds two doubles"},
  {"convgrid", GPUGrid_convgrid, METH_VARARGS, "Grid on the GPU"},
  {NULL, NULL}
};

/*
 * Python calls this to let us initialize our module
 */
void initGPUGrid()
{
  (void) Py_InitModule("GPUGrid", GPUGrid_methods);
}
}

