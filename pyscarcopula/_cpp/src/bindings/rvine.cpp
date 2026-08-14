#include "common.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_rvine(py::module_& m) {
    py::enum_<scar::RVineOpcode>(
        m, "RVineOpcode", "Opcodes used by packed R-vine programs.")
        .value("H", scar::RVineOpcode::H)
        .value("H_PAIR", scar::RVineOpcode::H_PAIR)
        .value("H_INV", scar::RVineOpcode::H_INV)
        .value("COPY", scar::RVineOpcode::COPY);

    py::enum_<scar::RVineNodeSource>(
        m, "RVineNodeSource", "Origin of a packed conditional-plan node.")
        .value("COMPUTED", scar::RVineNodeSource::Computed)
        .value("GIVEN", scar::RVineNodeSource::Given)
        .value("UNIFORM", scar::RVineNodeSource::Uniform);

    py::enum_<scar::rvine::ParameterSource>(
        m,
        "RVineParameterSource",
        "Storage location for one R-vine edge parameter.")
        .value("NONE", scar::rvine::ParameterSource::None)
        .value("SCALAR", scar::rvine::ParameterSource::Scalar)
        .value("ROW_PATH", scar::rvine::ParameterSource::RowPath);

    py::class_<scar::rvine::EdgeSpec>(
        m, "RVineEdgeSpec", "Model-independent native R-vine edge metadata.")
        .def(py::init<>())
        .def_readwrite("copula", &scar::rvine::EdgeSpec::copula)
        .def_readwrite(
            "parameter_source",
            &scar::rvine::EdgeSpec::parameter_source)
        .def_readwrite(
            "parameter_index", &scar::rvine::EdgeSpec::parameter_index)
        .def_readwrite(
            "parameter_free", &scar::rvine::EdgeSpec::parameter_free);

    py::class_<scar::RVineTraversalPlan>(
        m,
        "RVineTraversalPlan",
        "Model-independent execution plan for R-vine sampling.")
        .def(py::init<>())
        .def_readwrite("dimension", &scar::RVineTraversalPlan::dimension)
        .def_readwrite("node_count", &scar::RVineTraversalPlan::node_count)
        .def_readwrite(
            "last_uniform_column",
            &scar::RVineTraversalPlan::last_uniform_column)
        .def_readwrite(
            "last_output_node",
            &scar::RVineTraversalPlan::last_output_node)
        .def_readwrite(
            "output_nodes", &scar::RVineTraversalPlan::output_nodes)
        .def_readwrite(
            "column_uniforms", &scar::RVineTraversalPlan::column_uniforms)
        .def_readwrite(
            "inverse_offsets", &scar::RVineTraversalPlan::inverse_offsets)
        .def_readwrite(
            "inverse_edges", &scar::RVineTraversalPlan::inverse_edges)
        .def_readwrite(
            "inverse_partner_nodes",
            &scar::RVineTraversalPlan::inverse_partner_nodes)
        .def_readwrite(
            "inverse_output_nodes",
            &scar::RVineTraversalPlan::inverse_output_nodes)
        .def_readwrite(
            "inverse_transposed",
            &scar::RVineTraversalPlan::inverse_transposed)
        .def_readwrite(
            "forward_offsets", &scar::RVineTraversalPlan::forward_offsets)
        .def_readwrite(
            "forward_edges", &scar::RVineTraversalPlan::forward_edges)
        .def_readwrite(
            "forward_leaf_nodes",
            &scar::RVineTraversalPlan::forward_leaf_nodes)
        .def_readwrite(
            "forward_partner_nodes",
            &scar::RVineTraversalPlan::forward_partner_nodes)
        .def_readwrite(
            "forward_leaf_output_nodes",
            &scar::RVineTraversalPlan::forward_leaf_output_nodes)
        .def_readwrite(
            "forward_partner_output_nodes",
            &scar::RVineTraversalPlan::forward_partner_output_nodes)
        .def_readwrite(
            "forward_transposed",
            &scar::RVineTraversalPlan::forward_transposed)
        .def_readwrite(
            "update_u1_nodes",
            &scar::RVineTraversalPlan::update_u1_nodes)
        .def_readwrite(
            "update_u2_nodes",
            &scar::RVineTraversalPlan::update_u2_nodes);

    // Compatibility alias for callers that used the original GAS-specific
    // class name before the plan became model-independent.
    m.attr("GasRvinePlan") = m.attr("RVineTraversalPlan");

    py::class_<scar::RVineConditionalPlan>(
        m,
        "RVineConditionalPlan",
        "Flat native program for suffix or DAG conditional sampling.")
        .def(py::init<>())
        .def_readwrite("dimension", &scar::RVineConditionalPlan::dimension)
        .def_readwrite("node_count", &scar::RVineConditionalPlan::node_count)
        .def_readwrite(
            "given_variables", &scar::RVineConditionalPlan::given_variables)
        .def_readwrite("given_nodes", &scar::RVineConditionalPlan::given_nodes)
        .def_readwrite(
            "uniform_nodes", &scar::RVineConditionalPlan::uniform_nodes)
        .def_readwrite(
            "node_sources", &scar::RVineConditionalPlan::node_sources)
        .def_readwrite(
            "node_source_indices",
            &scar::RVineConditionalPlan::node_source_indices)
        .def_readwrite("opcodes", &scar::RVineConditionalPlan::opcodes)
        .def_readwrite(
            "edge_indices", &scar::RVineConditionalPlan::edge_indices)
        .def_readwrite(
            "input1_nodes", &scar::RVineConditionalPlan::input1_nodes)
        .def_readwrite(
            "input2_nodes", &scar::RVineConditionalPlan::input2_nodes)
        .def_readwrite(
            "output1_nodes", &scar::RVineConditionalPlan::output1_nodes)
        .def_readwrite(
            "output2_nodes", &scar::RVineConditionalPlan::output2_nodes)
        .def_readwrite(
            "transposed", &scar::RVineConditionalPlan::transposed)
        .def_readwrite(
            "output_nodes", &scar::RVineConditionalPlan::output_nodes)
        .def_readwrite("used_edges", &scar::RVineConditionalPlan::used_edges);

    py::class_<scar::RVineDensityPlan>(
        m,
        "RVineDensityPlan",
        "Flat native traversal for R-vine density and residual extraction.")
        .def(py::init<>())
        .def_readwrite("dimension", &scar::RVineDensityPlan::dimension)
        .def_readwrite("node_count", &scar::RVineDensityPlan::node_count)
        .def_readwrite("input_nodes", &scar::RVineDensityPlan::input_nodes)
        .def_readwrite("edge_indices", &scar::RVineDensityPlan::edge_indices)
        .def_readwrite("input1_nodes", &scar::RVineDensityPlan::input1_nodes)
        .def_readwrite("input2_nodes", &scar::RVineDensityPlan::input2_nodes)
        .def_readwrite("output1_nodes", &scar::RVineDensityPlan::output1_nodes)
        .def_readwrite("output2_nodes", &scar::RVineDensityPlan::output2_nodes)
        .def_readwrite("transposed", &scar::RVineDensityPlan::transposed)
        .def_readwrite(
            "residual_nodes", &scar::RVineDensityPlan::residual_nodes)
        .def_readwrite("used_edges", &scar::RVineDensityPlan::used_edges);

    m.def(
        "validate_rvine_conditional_plan",
        &scar::rvine::validate_conditional_plan,
        py::arg("plan"),
        py::arg("edge_count"));
    m.def(
        "validate_rvine_density_plan",
        &scar::rvine::validate_density_plan,
        py::arg("plan"),
        py::arg("edge_count"));
}

}  // namespace pyscarcopula::bindings
