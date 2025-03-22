#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "linear_regression.hpp"

namespace py = pybind11;

// Explicit instantiation for double type
template class LinearRegression<double>;

PYBIND11_MODULE(linear_regression, m) {
    py::class_<LinearRegression<double>>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression<double>::fit,
             py::arg("x"), py::arg("y"), 
             py::arg("learning_rate"), py::arg("epochs"))
        .def("predict", &LinearRegression<double>::predict)
        .def_property_readonly("getSlope", &LinearRegression<double>::getSlope)
        .def_property_readonly("getIntercept", &LinearRegression<double>::getIntercept);
}