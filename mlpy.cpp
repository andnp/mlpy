#include <boost/python.hpp>
#include <string>
#include <vector>
#include "Numpy2Eigen/Numpy2Eigen.hpp"
#include "gpucompute.hpp"
namespace py = boost::python;

class PyData : public Data {
 public:
    PyData(PyObject* data, PyObject* targets) : Data() {
        train_data = Numpy2Eigen::getEigenMatrix(data).transpose().cast<float>();
        train_targets = Numpy2Eigen::getEigenMatrix(targets).transpose().cast<float>();
    }

    void scaleData() {
        scale_data = 1;
        Data::scaleData();
    }

    Eigen::MatrixXf scaleData(Eigen::MatrixXf &X) {  // NOLINT
        if (scaled == 1) {
            Data::rowScale(X);
        }
        return X;
    }

    void printData() {
        std::cout << train_data << std::endl;
    }

 private:
    int scale_data = 0;
};

py::object scaleData(PyData& dat, PyObject* npX) {  // NOLINT
    Eigen::MatrixXf X = Numpy2Eigen::getEigenMatrix(npX).transpose().cast<float>();
    X = dat.scaleData(X).transpose();
    return Numpy2Eigen::getNumpyMatrix(X);
}

py::object predict(Algorithm& alg, PyObject* npX) {  // NOLINT
    Eigen::MatrixXf X = Numpy2Eigen::getEigenMatrix(npX).transpose().cast<float>();
    Eigen::MatrixXf P = alg.predict(X).transpose();
    return Numpy2Eigen::getNumpyMatrix(P);
}

void loadParameters(Algorithm& alg, py::dict &dict) {  // NOLINT
    json params;
    py::list keys = dict.keys();
    int n = py::len(keys);
    for (int i = 0; i < n; ++i) {
        py::object k = keys[i];
        py::object val = dict.get(k);
        std::string type = boost::python::extract<std::string>(val.attr("__class__").attr("__name__"));
        std::string key = py::extract<std::string>(py::str(k))();
        if (type == "float") {
            params[key] = py::extract<float>(val)();
        } else if (type == "int") {
            params[key] = py::extract<int>(val)();
        } else if (type == "str") {
            params[key] = py::extract<std::string>(py::str(val))();
        } else if (type == "list") {
            std::vector<float> vec;
            for (int j = 0; j < py::len(val); ++j) {
                vec.push_back(py::extract<float>(val[i])());
            }
            params[key] = vec;
        }
    }
    alg.loadParameters(params);
}

BOOST_PYTHON_MODULE(mlpy) {
    import_array();
    Numpy2Eigen::init();
    Logger::setInfo(Output::stdout);
    py::class_<Algorithm, boost::noncopyable>("Algorithm", py::no_init)
        .def("printHeader", &Algorithm::printHeader)
        .def("loadData", static_cast<void(Algorithm::*)(Data&)>(&Algorithm::loadData))
        .def("optimizeBatch", &Algorithm::optimizeBatch)
        .def("reset", &Algorithm::reset)
        .def("predict", predict)
        .def("loadParameters", loadParameters)
        .def("printParameters", &Algorithm::print);

    py::class_<ANN, py::bases<Algorithm>>("ANN", py::init<int>());

    py::class_<Data, boost::noncopyable>("AbsData", py::no_init);
    py::class_<PyData, py::bases<Data>>("Data", py::init<PyObject*, PyObject*>())
        .def("scaleData", static_cast<void(PyData::*)(void)>(&PyData::scaleData))
        .def("scaleExternal", scaleData)
        .def("printData", &PyData::printData);
}
