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
        .def_readwrite("used_edges", &scar::RVineDensityPlan::used_edges)
        .def_readwrite(
            "affected_operation_offsets",
            &scar::RVineDensityPlan::affected_operation_offsets)
        .def_readwrite(
            "affected_operations",
            &scar::RVineDensityPlan::affected_operations)
        .def_readwrite(
            "affected_node_offsets",
            &scar::RVineDensityPlan::affected_node_offsets)
        .def_readwrite(
            "affected_nodes", &scar::RVineDensityPlan::affected_nodes);

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

    m.def(
        "rvine_sample",
        [](const scar::RVineTraversalPlan& plan,
           const std::vector<scar::rvine::EdgeSpec>& edges,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> scalar_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> row_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> uniforms,
           int n_threads) {
            const py::buffer_info scalar_info = scalar_parameters.request();
            const py::buffer_info row_info = row_parameters.request();
            const py::buffer_info uniform_info = uniforms.request();
            if (scalar_info.ndim != 1 || row_info.ndim != 2
                || uniform_info.ndim != 2) {
                throw std::invalid_argument(
                    "scalar_parameters must be 1D and row_parameters and "
                    "uniforms must be 2D arrays");
            }
            scar::rvine::ParameterPack parameters;
            parameters.scalar_parameters = {
                static_cast<const double*>(scalar_info.ptr),
                static_cast<std::size_t>(scalar_info.shape[0]),
            };
            parameters.row_parameters = {
                static_cast<const double*>(row_info.ptr),
                static_cast<std::size_t>(row_info.size),
            };
            parameters.n_rows = static_cast<std::int64_t>(row_info.shape[0]);
            parameters.row_parameter_columns =
                static_cast<std::int64_t>(row_info.shape[1]);
            const scar::DoubleView uniform_view = {
                static_cast<const double*>(uniform_info.ptr),
                static_cast<std::size_t>(uniform_info.size),
            };
            scar::rvine::SampleResult result;
            {
                py::gil_scoped_release release;
                result = scar::rvine::sample(
                    plan,
                    edges,
                    parameters,
                    uniform_view,
                    static_cast<std::int64_t>(uniform_info.shape[0]),
                    static_cast<std::int64_t>(uniform_info.shape[1]),
                    n_threads);
            }
            py::dict diagnostics;
            diagnostics["n_threads_requested"] = result.n_threads_requested;
            diagnostics["n_threads_used"] = result.n_threads_used;
            diagnostics["inverse_operations"] = result.inverse_operations;
            diagnostics["forward_operations"] = result.forward_operations;
            diagnostics["independence_fast_paths"] =
                result.independence_fast_paths;

            py::dict out;
            out["values"] = vector_to_array(result.values);
            out["n_rows"] = result.n_rows;
            out["dimension"] = result.dimension;
            out["status"] = result.status;
            out["failure_row"] = result.failure_row;
            out["failure_edge"] = result.failure_edge;
            out["failure_operation"] = result.failure_operation;
            out["diagnostics"] = std::move(diagnostics);
            return out;
        },
        py::arg("plan"),
        py::arg("edges"),
        py::arg("scalar_parameters"),
        py::arg("row_parameters"),
        py::arg("uniforms"),
        py::arg("n_threads") = 1);

    m.def(
        "rvine_conditional_sample",
        [](const scar::RVineConditionalPlan& plan,
           const std::vector<scar::rvine::EdgeSpec>& edges,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> scalar_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> row_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> given_values,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> uniforms,
           int n_threads) {
            const py::buffer_info scalar_info = scalar_parameters.request();
            const py::buffer_info row_info = row_parameters.request();
            const py::buffer_info given_info = given_values.request();
            const py::buffer_info uniform_info = uniforms.request();
            if (scalar_info.ndim != 1 || row_info.ndim != 2
                || given_info.ndim != 1 || uniform_info.ndim != 2) {
                throw std::invalid_argument(
                    "scalar_parameters and given_values must be 1D and "
                    "row_parameters and uniforms must be 2D arrays");
            }
            scar::rvine::ParameterPack parameters;
            parameters.scalar_parameters = {
                static_cast<const double*>(scalar_info.ptr),
                static_cast<std::size_t>(scalar_info.shape[0]),
            };
            parameters.row_parameters = {
                static_cast<const double*>(row_info.ptr),
                static_cast<std::size_t>(row_info.size),
            };
            parameters.n_rows = static_cast<std::int64_t>(row_info.shape[0]);
            parameters.row_parameter_columns =
                static_cast<std::int64_t>(row_info.shape[1]);
            const scar::DoubleView given_view = {
                static_cast<const double*>(given_info.ptr),
                static_cast<std::size_t>(given_info.size),
            };
            const scar::DoubleView uniform_view = {
                static_cast<const double*>(uniform_info.ptr),
                static_cast<std::size_t>(uniform_info.size),
            };
            scar::rvine::ConditionalSampleResult result;
            {
                py::gil_scoped_release release;
                result = scar::rvine::conditional_sample(
                    plan,
                    edges,
                    parameters,
                    given_view,
                    uniform_view,
                    static_cast<std::int64_t>(uniform_info.shape[0]),
                    static_cast<std::int64_t>(uniform_info.shape[1]),
                    n_threads);
            }
            py::dict diagnostics;
            diagnostics["n_threads_requested"] = result.n_threads_requested;
            diagnostics["n_threads_used"] = result.n_threads_used;
            diagnostics["h_operations"] = result.h_operations;
            diagnostics["h_pair_operations"] = result.h_pair_operations;
            diagnostics["inverse_operations"] = result.inverse_operations;
            diagnostics["copy_operations"] = result.copy_operations;
            diagnostics["independence_fast_paths"] =
                result.independence_fast_paths;
            diagnostics["row_blocks"] = result.row_blocks;
            diagnostics["max_block_rows"] = result.max_block_rows;
            diagnostics["peak_workspace_bytes"] =
                result.peak_workspace_bytes;

            py::dict out;
            out["values"] = vector_to_array(result.values);
            out["n_rows"] = result.n_rows;
            out["dimension"] = result.dimension;
            out["status"] = result.status;
            out["failure_row"] = result.failure_row;
            out["failure_edge"] = result.failure_edge;
            out["failure_operation"] = result.failure_operation;
            out["diagnostics"] = std::move(diagnostics);
            return out;
        },
        py::arg("plan"),
        py::arg("edges"),
        py::arg("scalar_parameters"),
        py::arg("row_parameters"),
        py::arg("given_values"),
        py::arg("uniforms"),
        py::arg("n_threads") = 1);

    m.def(
        "rvine_log_pdf_rows",
        [](const scar::RVineDensityPlan& plan,
           const std::vector<scar::rvine::EdgeSpec>& edges,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> scalar_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> row_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> observations,
           int n_threads) {
            const py::buffer_info scalar_info = scalar_parameters.request();
            const py::buffer_info row_info = row_parameters.request();
            const py::buffer_info observation_info = observations.request();
            if (scalar_info.ndim != 1 || row_info.ndim != 2
                || observation_info.ndim != 2) {
                throw std::invalid_argument(
                    "scalar_parameters must be 1D and row_parameters and "
                    "observations must be 2D arrays");
            }
            scar::rvine::ParameterPack parameters;
            parameters.scalar_parameters = {
                static_cast<const double*>(scalar_info.ptr),
                static_cast<std::size_t>(scalar_info.shape[0]),
            };
            parameters.row_parameters = {
                static_cast<const double*>(row_info.ptr),
                static_cast<std::size_t>(row_info.size),
            };
            parameters.n_rows = static_cast<std::int64_t>(row_info.shape[0]);
            parameters.row_parameter_columns =
                static_cast<std::int64_t>(row_info.shape[1]);
            const scar::DoubleView observation_view = {
                static_cast<const double*>(observation_info.ptr),
                static_cast<std::size_t>(observation_info.size),
            };
            scar::rvine::DensityResult result;
            {
                py::gil_scoped_release release;
                result = scar::rvine::log_pdf_rows(
                    plan,
                    edges,
                    parameters,
                    observation_view,
                    static_cast<std::int64_t>(observation_info.shape[0]),
                    static_cast<std::int64_t>(observation_info.shape[1]),
                    n_threads);
            }
            py::dict diagnostics;
            diagnostics["n_threads_requested"] = result.n_threads_requested;
            diagnostics["n_threads_used"] = result.n_threads_used;
            diagnostics["density_operations"] =
                result.diagnostics.density_operations;
            diagnostics["h_pair_operations"] =
                result.diagnostics.h_pair_operations;
            diagnostics["independence_fast_paths"] =
                result.diagnostics.independence_fast_paths;

            py::dict out;
            out["log_pdf"] = vector_to_array(result.log_pdf);
            out["n_rows"] = result.n_rows;
            out["dimension"] = result.dimension;
            out["status"] = result.status;
            out["failure_row"] = result.failure_row;
            out["failure_edge"] = result.failure_edge;
            out["failure_operation"] = result.failure_operation;
            out["diagnostics"] = std::move(diagnostics);
            return out;
        },
        py::arg("plan"),
        py::arg("edges"),
        py::arg("scalar_parameters"),
        py::arg("row_parameters"),
        py::arg("observations"),
        py::arg("n_threads") = 1);

    m.def(
        "rvine_rosenblatt_transform",
        [](const scar::RVineDensityPlan& plan,
           const std::vector<scar::rvine::EdgeSpec>& edges,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> scalar_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> row_parameters,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> observations,
           int n_threads) {
            const py::buffer_info scalar_info = scalar_parameters.request();
            const py::buffer_info row_info = row_parameters.request();
            const py::buffer_info observation_info = observations.request();
            if (scalar_info.ndim != 1 || row_info.ndim != 2
                || observation_info.ndim != 2) {
                throw std::invalid_argument(
                    "scalar_parameters must be 1D and row_parameters and "
                    "observations must be 2D arrays");
            }
            scar::rvine::ParameterPack parameters;
            parameters.scalar_parameters = {
                static_cast<const double*>(scalar_info.ptr),
                static_cast<std::size_t>(scalar_info.shape[0]),
            };
            parameters.row_parameters = {
                static_cast<const double*>(row_info.ptr),
                static_cast<std::size_t>(row_info.size),
            };
            parameters.n_rows = static_cast<std::int64_t>(row_info.shape[0]);
            parameters.row_parameter_columns =
                static_cast<std::int64_t>(row_info.shape[1]);
            const scar::DoubleView observation_view = {
                static_cast<const double*>(observation_info.ptr),
                static_cast<std::size_t>(observation_info.size),
            };
            scar::rvine::RosenblattResult result;
            {
                py::gil_scoped_release release;
                result = scar::rvine::rosenblatt_transform(
                    plan,
                    edges,
                    parameters,
                    observation_view,
                    static_cast<std::int64_t>(observation_info.shape[0]),
                    static_cast<std::int64_t>(observation_info.shape[1]),
                    n_threads);
            }
            py::dict diagnostics;
            diagnostics["n_threads_requested"] = result.n_threads_requested;
            diagnostics["n_threads_used"] = result.n_threads_used;
            diagnostics["h_pair_operations"] = result.h_pair_operations;
            diagnostics["independence_fast_paths"] =
                result.independence_fast_paths;

            py::dict out;
            out["residuals"] = vector_to_array(result.residuals);
            out["n_rows"] = result.n_rows;
            out["dimension"] = result.dimension;
            out["status"] = result.status;
            out["failure_row"] = result.failure_row;
            out["failure_edge"] = result.failure_edge;
            out["failure_operation"] = result.failure_operation;
            out["diagnostics"] = std::move(diagnostics);
            return out;
        },
        py::arg("plan"),
        py::arg("edges"),
        py::arg("scalar_parameters"),
        py::arg("row_parameters"),
        py::arg("observations"),
        py::arg("n_threads") = 1);

    const auto mcmc_binding = [](
        const scar::RVineDensityPlan& plan,
        const std::vector<scar::rvine::EdgeSpec>& edges,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> scalar_parameters,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> row_parameters,
        const std::vector<int>& given_indices,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> given_values,
        const std::vector<int>& free_indices,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> current_state,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> current_log_pdf,
        std::int64_t global_step_offset,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> proposal_uniforms,
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast> acceptance_uniforms,
        int n_threads,
        const std::string& density_algorithm,
        std::uint64_t memory_budget_bytes) {
        const py::buffer_info scalar_info = scalar_parameters.request();
        const py::buffer_info row_info = row_parameters.request();
        const py::buffer_info given_info = given_values.request();
        const py::buffer_info state_info = current_state.request();
        const py::buffer_info log_pdf_info = current_log_pdf.request();
        const py::buffer_info proposal_info = proposal_uniforms.request();
        const py::buffer_info acceptance_info = acceptance_uniforms.request();
        if (scalar_info.ndim != 1 || row_info.ndim != 2
            || given_info.ndim != 1 || state_info.ndim != 2
            || log_pdf_info.ndim != 1 || proposal_info.ndim != 2
            || acceptance_info.ndim != 2) {
            throw std::invalid_argument(
                "R-vine MCMC scalar/given/log-density buffers must be 1D "
                "and row/state/draw buffers must be 2D arrays");
        }
        scar::rvine::ParameterPack parameters;
        parameters.scalar_parameters = {
            static_cast<const double*>(scalar_info.ptr),
            static_cast<std::size_t>(scalar_info.shape[0]),
        };
        parameters.row_parameters = {
            static_cast<const double*>(row_info.ptr),
            static_cast<std::size_t>(row_info.size),
        };
        parameters.n_rows = static_cast<std::int64_t>(row_info.shape[0]);
        parameters.row_parameter_columns =
            static_cast<std::int64_t>(row_info.shape[1]);
        const scar::DoubleView given_view = {
            static_cast<const double*>(given_info.ptr),
            static_cast<std::size_t>(given_info.size),
        };
        const scar::DoubleView state_view = {
            static_cast<const double*>(state_info.ptr),
            static_cast<std::size_t>(state_info.size),
        };
        const scar::DoubleView log_pdf_view = {
            static_cast<const double*>(log_pdf_info.ptr),
            static_cast<std::size_t>(log_pdf_info.size),
        };
        const scar::DoubleView proposal_view = {
            static_cast<const double*>(proposal_info.ptr),
            static_cast<std::size_t>(proposal_info.size),
        };
        const scar::DoubleView acceptance_view = {
            static_cast<const double*>(acceptance_info.ptr),
            static_cast<std::size_t>(acceptance_info.size),
        };
        scar::rvine::MCMCDensityAlgorithm native_algorithm;
        if (density_algorithm == "auto") {
            native_algorithm = scar::rvine::MCMCDensityAlgorithm::Auto;
        } else if (density_algorithm == "full_recompute") {
            native_algorithm =
                scar::rvine::MCMCDensityAlgorithm::FullRecompute;
        } else if (density_algorithm == "incremental") {
            native_algorithm = scar::rvine::MCMCDensityAlgorithm::Incremental;
        } else {
            throw std::invalid_argument(
                "R-vine MCMC density_algorithm must be 'auto', "
                "'full_recompute', or 'incremental'");
        }
        scar::rvine::MCMCResult result;
        {
            py::gil_scoped_release release;
            result = scar::rvine::mcmc_chunk(
                plan,
                edges,
                parameters,
                given_indices,
                given_view,
                free_indices,
                state_view,
                static_cast<std::int64_t>(state_info.shape[0]),
                static_cast<std::int64_t>(state_info.shape[1]),
                log_pdf_view,
                global_step_offset,
                proposal_view,
                static_cast<std::int64_t>(proposal_info.shape[0]),
                static_cast<std::int64_t>(proposal_info.shape[1]),
                acceptance_view,
                static_cast<std::int64_t>(acceptance_info.shape[0]),
                static_cast<std::int64_t>(acceptance_info.shape[1]),
                n_threads,
                native_algorithm,
                memory_budget_bytes);
        }
        py::dict diagnostics;
        diagnostics["n_threads_requested"] = result.n_threads_requested;
        diagnostics["n_threads_used"] = result.n_threads_used;
        diagnostics["proposed"] = result.proposed;
        diagnostics["accepted"] = result.accepted;
        diagnostics["non_finite_proposals"] =
            result.non_finite_proposals;
        diagnostics["mcmc_density_algorithm"] =
            result.density_algorithm
                == scar::rvine::MCMCDensityAlgorithm::Incremental
            ? "incremental"
            : "full_recompute";
        diagnostics["affected_operations"] = result.affected_operations;
        diagnostics["affected_operation_evaluations"] =
            result.affected_operation_evaluations;
        diagnostics["cache_bytes"] = result.cache_bytes;
        diagnostics["peak_workspace_bytes"] =
            result.peak_workspace_bytes;
        diagnostics["row_chunks"] = result.row_chunks;
        diagnostics["max_chunk_rows"] = result.max_chunk_rows;
        diagnostics["memory_budget_bytes"] = result.memory_budget_bytes;

        py::dict out;
        out["state"] = vector_to_array(result.state);
        out["log_pdf"] = vector_to_array(result.log_pdf);
        out["n_rows"] = result.n_rows;
        out["dimension"] = result.dimension;
        out["coordinate_steps"] = result.coordinate_steps;
        out["status"] = result.status;
        out["failure_row"] = result.failure_row;
        out["failure_edge"] = result.failure_edge;
        out["failure_operation"] = result.failure_operation;
        out["diagnostics"] = std::move(diagnostics);
        return out;
    };
    m.def(
        "rvine_mcmc",
        mcmc_binding,
        py::arg("plan"),
        py::arg("edges"),
        py::arg("scalar_parameters"),
        py::arg("row_parameters"),
        py::arg("given_indices"),
        py::arg("given_values"),
        py::arg("free_indices"),
        py::arg("current_state"),
        py::arg("current_log_pdf"),
        py::arg("global_step_offset"),
        py::arg("proposal_uniforms"),
        py::arg("acceptance_uniforms"),
        py::arg("n_threads") = 1,
        py::arg("density_algorithm") = "full_recompute",
        py::arg("memory_budget_bytes") = 64U * 1024U * 1024U);
    m.def(
        "rvine_mcmc_chunk",
        mcmc_binding,
        py::arg("plan"),
        py::arg("edges"),
        py::arg("scalar_parameters"),
        py::arg("row_parameters"),
        py::arg("given_indices"),
        py::arg("given_values"),
        py::arg("free_indices"),
        py::arg("current_state"),
        py::arg("current_log_pdf"),
        py::arg("global_step_offset"),
        py::arg("proposal_uniforms"),
        py::arg("acceptance_uniforms"),
        py::arg("n_threads") = 1,
        py::arg("density_algorithm") = "full_recompute",
        py::arg("memory_budget_bytes") = 64U * 1024U * 1024U);
}

}  // namespace pyscarcopula::bindings
