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
#include "libdl/variable.h"
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
    .def(py::init<std::shared_ptr<GraphNode>>())
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
    py::class_<GraphNode, std::shared_ptr<GraphNode>>GraphNode(m, "GraphNode");
            GraphNode.def("getName", &GraphNode::getName);
            GraphNode.def("getType", &GraphNode::getType);
            GraphNode.def("setData", &GraphNode::setData);
            GraphNode.def("getData", &GraphNode::getData);
            GraphNode.def("setGradient", &GraphNode::setGradient);

    py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable", GraphNode)
            .def(py::init<std::string, Tensor4f>());
    py::class_<Placeholder, std::shared_ptr<Placeholder>>(m, "Placeholder", GraphNode)
            .def(py::init<std::string, Tensor4f>());
    py::class_<MatrixMultiplication, std::shared_ptr<MatrixMultiplication>>(m, "MatrixMultiplication", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<ElementwiseAdd, std::shared_ptr<ElementwiseAdd>>(m, "ElementwiseAdd", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<Sigmoid, std::shared_ptr<Sigmoid>>(m, "Sigmoid", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<ReLU, std::shared_ptr<ReLU>>(m, "ReLU", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<Conv2d, std::shared_ptr<Conv2d>>(m, "Conv2d", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<Pool_average, std::shared_ptr<Pool_average>>(m, "Pool_average", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<Softmax, std::shared_ptr<Softmax>>(m, "Softmax", GraphNode)
            .def(py::init<std::string, NodeVec>());
    py::class_<TanH, std::shared_ptr<TanH>>(m, "TanH", GraphNode)
            .def(py::init<std::string, NodeVec>());


// load store
m.def("save_weights", &save_weights);
m.def("load_weights", &load_weights);
}