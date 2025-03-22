#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dtc.hpp"

namespace py = pybind11;

template class decision_tree_classifier<double>;

PYBIND11_MODULE(dtc, m) {
    py::class_<decision_tree_classifier<double>>(m, "DecisionTreeClassifier")
        .def(py::init<int>(), py::arg("max_depth") = -1)
        .def("fit", &decision_tree_classifier<double>::fit)
        .def("predict", &decision_tree_classifier<double>::predict);
}