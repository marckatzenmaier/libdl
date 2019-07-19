//
// Created by marc on 26.06.19.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "libdl/graph.h"
#include "libdl/helper_functions.h"
#include "libdl/loss.h"
#include "libdl/Optimizer.h"
#include "libdl/graph_node.h"
#include "libdl/Variable.h"
#include "libdl/initializer.h"
#include <iostream>

namespace py = pybind11;

namespace pybind11 { namespace detail {
        template <> struct type_caster<Tensor4f> {
        public:
            /**
             * This macro establishes the name 'inty' in
             * function signatures and declares a local variable
             * 'value' of type inty
             */
        PYBIND11_TYPE_CASTER(Tensor4f, _("Tensor4f"));

            /**
             * Conversion part 1 (Python->C++): convert a PyObject into a inty
             * instance or return false upon failure. The second argument
             * indicates whether implicit conversions should be applied.
             */

            bool load(handle src, bool) {
                if (!isinstance<array_t<float>>(src))
                    return false;
                auto buf = array::ensure(src);

                if (!buf)
                    return false;

                auto dims = buf.ndim();
                if (dims != 4)
                    return false;

                value = Eigen::TensorMap<Tensor4f>((float*)buf.data(), buf.shape(0), buf.shape(1), buf.shape(2), buf.shape(3));
                return true;
            }

            /**
             * Conversion part 2 (C++ -> Python): convert an inty instance into
             * a Python object. The second and third arguments are used to
             * indicate the return value policy and parent object (for
             * ``return_value_policy::reference_internal``) and are generally
             * ignored by implicit casters.
             */
            static handle cast(Tensor4f src, return_value_policy /*policy*/, handle /* parent */) {
                detail::any_container<ssize_t> shape({src.dimension(0), src.dimension(1), src.dimension(2), src.dimension(3)});
                const py::dtype dt("float32");
                return py::array(dt, shape, src.data()).release();
            }

        };
    }}

PYBIND11_MODULE(my_dllib_py, m) {
//Tensor wrap

//Graph
py::class_<Graph, std::unique_ptr<Graph>>graph(m, "Graph");
graph
    .def("forward", &Graph::forward)
    .def("backward",&Graph::backward)
    .def("setPlaceholder", &Graph::setPlaceholder)
    .def("getWeights", &Graph::getWeights)
    .def("getEndpoint", &Graph::getEndpoint)
    .def("clearGradients", &Graph::clearGradients);


// losses
m.def("loss_MSE", &loss_MSE, py::arg("output"), py::arg("label"));
m.def("loss_Crossentropy", &loss_Crossentropy, py::arg("output"), py::arg("label"));


// optim
py::class_<SGD_Optimizer>sgd_Optimizer(m, "SGD_Optimizer");
sgd_Optimizer
    .def("optimize", &SGD_Optimizer::optimize)
    .def(py::init<std::vector<std::shared_ptr<Variable>>, float>());

// init weights
m.def("init_weights_random", &init_weights_random);

//net generation
m.def("make_LeNet", &make_LeNet, py::arg("batch_size"));
m.def("make_LeNet_siamnese", &make_LeNet_siamnese);



//Graph_node
py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable");
py::class_<GraphNode, std::shared_ptr<GraphNode>>(m, "GraphNode")
    .def(py::init<std::string>())
        .def("getName", &GraphNode::getName)
        .def("getType", &GraphNode::getType)
        .def("setData", &GraphNode::setData)
        .def("getData", &GraphNode::getData)
        .def("setGradient", &GraphNode::setGradient);

// load store
m.def("save_weights", &save_weights);
m.def("load_weights", &load_weights);
}