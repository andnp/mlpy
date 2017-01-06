#include <boost/python.hpp>

char const* greet() {
    return "hello, world";
}

BOOST_PYTHON_MODULE(hello) {
    boost::python::def("greet", greet);
}
