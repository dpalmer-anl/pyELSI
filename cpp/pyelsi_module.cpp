#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>

#include <cstring>
#include <complex>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include "elsi.h"

namespace py = pybind11;

extern "C" {
int Csys2blacs_handle(MPI_Comm comm);
void Cblacs_get(int context, int request, int* value);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
}

static py::dict build_info_dict() {
  py::dict d;
  d["has_mpi"] = true;

#if PYELSI_HAS_CUDA
  d["has_cuda"] = true;
#else
  d["has_cuda"] = false;
#endif
  d["backend"] = "elsi";
  return d;
}

static void maybe_mpi_init() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided = 0;
    // Many ELSI backends do not require MPI_THREAD_MULTIPLE and some vendored
    // dependencies are less stable under it. Request a conservative level.
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
  }
}

static int blacs_ctxt_1x1(MPI_Comm comm) {
  // Tie BLACS context to the provided communicator. This matters under mpirun
  // when creating a per-rank 1x1 context (MPI_COMM_SELF).
  int ictxt = Csys2blacs_handle(comm);
  Cblacs_gridinit(&ictxt, "r", 1, 1);
  return ictxt;
}

static void apply_options(elsi_handle h, const std::map<std::string, py::object>& opts) {
  auto get_int = [&](const char* k) -> std::optional<int> {
    auto it = opts.find(k);
    if (it == opts.end()) return std::nullopt;
    return it->second.cast<int>();
  };
  auto get_double = [&](const char* k) -> std::optional<double> {
    auto it = opts.find(k);
    if (it == opts.end()) return std::nullopt;
    return it->second.cast<double>();
  };

  // General output / overlap settings
  if (auto v = get_int("output")) c_elsi_set_output(h, *v);
  if (auto v = get_int("output_log")) c_elsi_set_output_log(h, *v);
  if (auto v = get_int("unit_ovlp")) c_elsi_set_unit_ovlp(h, *v);

  // Spectrum bounds (required by NTPoly; useful for PEXSI chemical-potential search)
  if (auto v = get_double("energy_gap")) c_elsi_set_energy_gap(h, *v);
  if (auto v = get_double("spectrum_width")) c_elsi_set_spectrum_width(h, *v);

  // Chemical-potential broadening (shared across DM solvers)
  if (auto v = get_double("mu_broaden_width")) c_elsi_set_mu_broaden_width(h, *v);
  if (auto v = get_int("mu_broaden_scheme")) c_elsi_set_mu_broaden_scheme(h, *v);

  // ELPA settings
  if (auto v = get_int("elpa_solver")) c_elsi_set_elpa_solver(h, *v);
  if (auto v = get_int("elpa_gpu")) c_elsi_set_elpa_gpu(h, *v);
  if (auto v = get_int("elpa_n_single")) c_elsi_set_elpa_n_single(h, *v);
  if (auto v = get_int("elpa_autotune")) c_elsi_set_elpa_autotune(h, *v);

  // OMM settings
  if (auto v = get_int("omm_flavor")) c_elsi_set_omm_flavor(h, *v);
  if (auto v = get_int("omm_n_elpa")) c_elsi_set_omm_n_elpa(h, *v);
  if (auto v = get_double("omm_tol")) c_elsi_set_omm_tol(h, *v);

  // PEXSI settings
  if (auto v = get_int("pexsi_np_per_pole")) c_elsi_set_pexsi_np_per_pole(h, *v);
  if (auto v = get_int("pexsi_n_mu")) c_elsi_set_pexsi_n_mu(h, *v);
  if (auto v = get_int("pexsi_n_pole")) c_elsi_set_pexsi_n_pole(h, *v);
  if (auto v = get_double("pexsi_temp")) c_elsi_set_pexsi_temp(h, *v);
  if (auto v = get_double("pexsi_delta_e")) c_elsi_set_pexsi_delta_e(h, *v);
  if (auto v = get_double("pexsi_mu_min")) c_elsi_set_pexsi_mu_min(h, *v);
  if (auto v = get_double("pexsi_mu_max")) c_elsi_set_pexsi_mu_max(h, *v);
  if (auto v = get_double("pexsi_gap")) c_elsi_set_pexsi_gap(h, *v);
  if (auto v = get_double("pexsi_inertia_tol")) c_elsi_set_pexsi_inertia_tol(h, *v);

  // NTPoly settings (all required to avoid segfault; caller must set spectrum_width)
  if (auto v = get_int("ntpoly_method")) c_elsi_set_ntpoly_method(h, *v);
  if (auto v = get_int("ntpoly_isr")) c_elsi_set_ntpoly_isr(h, *v);
  if (auto v = get_double("ntpoly_tol")) c_elsi_set_ntpoly_tol(h, *v);
  if (auto v = get_double("ntpoly_filter")) c_elsi_set_ntpoly_filter(h, *v);
  if (auto v = get_int("ntpoly_max_iter")) c_elsi_set_ntpoly_max_iter(h, *v);

  // ChASE settings
  if (auto v = get_double("chase_tol")) c_elsi_set_chase_tol(h, *v);
}

/** ELSI C bindings always c_f_pointer(ovlp); NULL crashes. Provide a dummy buffer when S is omitted. */
static py::array_t<double, py::array::f_style> identity_fortran_colmajor(int n) {
  py::array_t<double, py::array::f_style> Id({n, n});
  double* p = static_cast<double*>(Id.mutable_data());
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::memset(p, 0, nn * sizeof(double));
  for (int j = 0; j < n; ++j) {
    p[j * n + j] = 1.0;
  }
  return Id;
}

static py::tuple elsi_ev_real_dense(py::array_t<double, py::array::f_style | py::array::forcecast> H,
                                    py::object S_obj, int solver, double n_electron, int n_state,
                                    py::dict backend_opts, bool want_vectors) {
  maybe_mpi_init();

  if (H.ndim() != 2 || H.shape(0) != H.shape(1)) throw std::runtime_error("H must be a square 2D array");
  const int n_basis = static_cast<int>(H.shape(0));

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  py::array_t<double, py::array::f_style | py::array::forcecast> S;
  py::array_t<double, py::array::f_style> S_dummy;
  double* ovlp = nullptr;

  // Create handle
  elsi_handle h = nullptr;
  // Default: MULTI_PROC so ELSI initializes MPI/BLACS metadata.
  // For MPI unit tests we allow an override to run per-rank serial by using
  // MPI_COMM_SELF (size=1) while still using MULTI_PROC mode.
  c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
  const MPI_Comm elsi_comm_ev = force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD;
  c_elsi_set_mpi(h, MPI_Comm_c2f(elsi_comm_ev));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt = blacs_ctxt_1x1(elsi_comm_ev);
  c_elsi_set_blacs(h, ictxt, 32);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  if (!S_obj.is_none()) {
    S = S_obj.cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    if (S.ndim() != 2 || S.shape(0) != n_basis || S.shape(1) != n_basis) throw std::runtime_error("S must be (n,n)");
    ovlp = static_cast<double*>(S.mutable_data());
    c_elsi_set_unit_ovlp(h, 0);
  } else {
    S_dummy = identity_fortran_colmajor(n_basis);
    ovlp = static_cast<double*>(S_dummy.mutable_data());
    c_elsi_set_unit_ovlp(h, 1);
  }

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    auto k = py::reinterpret_borrow<py::str>(item.first);
    auto v = py::reinterpret_borrow<py::object>(item.second);
    opts[k.cast<std::string>()] = std::move(v);
  }
  apply_options(h, opts);

  py::array_t<double> eval({n_basis});
  py::array_t<double, py::array::f_style> evec({n_basis, n_basis});

  double* ham = static_cast<double*>(H.mutable_data());
  double* ev = static_cast<double*>(eval.mutable_data());
  double* vec = want_vectors ? static_cast<double*>(evec.mutable_data()) : nullptr;

  if (want_vectors) {
    c_elsi_ev_real(h, ham, ovlp, ev, vec);
  } else {
    // ELSI API always takes evec pointer; pass allocated but ignore
    c_elsi_ev_real(h, ham, ovlp, ev, static_cast<double*>(evec.mutable_data()));
  }

  c_elsi_finalize(h);

  if (!want_vectors) return py::make_tuple(eval, py::none());
  return py::make_tuple(eval, evec);
}

static py::tuple elsi_dm_real_dense(py::array_t<double, py::array::f_style | py::array::forcecast> H,
                                    py::object S_obj, int solver, double n_electron, int n_state, py::dict backend_opts) {
  maybe_mpi_init();
  if (H.ndim() != 2 || H.shape(0) != H.shape(1)) throw std::runtime_error("H must be a square 2D array");
  const int n_basis = static_cast<int>(H.shape(0));

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  py::array_t<double, py::array::f_style | py::array::forcecast> S;
  py::array_t<double, py::array::f_style> S_dummy;
  double* ovlp = nullptr;

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
  const MPI_Comm elsi_comm_dm = force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD;
  c_elsi_set_mpi(h, MPI_Comm_c2f(elsi_comm_dm));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt_dm = blacs_ctxt_1x1(elsi_comm_dm);
  c_elsi_set_blacs(h, ictxt_dm, 32);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  if (!S_obj.is_none()) {
    S = S_obj.cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    if (S.ndim() != 2 || S.shape(0) != n_basis || S.shape(1) != n_basis) throw std::runtime_error("S must be (n,n)");
    ovlp = static_cast<double*>(S.mutable_data());
    c_elsi_set_unit_ovlp(h, 0);
  } else {
    S_dummy = identity_fortran_colmajor(n_basis);
    ovlp = static_cast<double*>(S_dummy.mutable_data());
    c_elsi_set_unit_ovlp(h, 1);
  }

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    auto k = py::reinterpret_borrow<py::str>(item.first);
    auto v = py::reinterpret_borrow<py::object>(item.second);
    opts[k.cast<std::string>()] = std::move(v);
  }
  apply_options(h, opts);

  py::array_t<double, py::array::f_style> dm({n_basis, n_basis});
  py::array_t<double> energy({1});

  double* ham = static_cast<double*>(H.mutable_data());
  double* dm_ptr = static_cast<double*>(dm.mutable_data());
  double* e_ptr = static_cast<double*>(energy.mutable_data());

  c_elsi_dm_real(h, ham, ovlp, dm_ptr, e_ptr);
  c_elsi_finalize(h);

  return py::make_tuple(dm, energy.at(0));
}

static py::tuple elsi_dm_real_csc(py::array_t<double, py::array::c_style | py::array::forcecast> ham_val,
                                  py::array_t<int, py::array::c_style | py::array::forcecast> row_ind_1based,
                                  py::array_t<int, py::array::c_style | py::array::forcecast> col_ptr_1based,
                                  int n_basis, py::object ovlp_val_obj, int solver, double n_electron, int n_state,
                                  py::dict backend_opts) {
  maybe_mpi_init();

  if (ham_val.ndim() != 1) throw std::runtime_error("ham_val must be a 1D array (CSC values)");
  if (row_ind_1based.ndim() != 1) throw std::runtime_error("row_ind must be 1D");
  if (col_ptr_1based.ndim() != 1) throw std::runtime_error("col_ptr must be 1D");
  if (col_ptr_1based.shape(0) != static_cast<py::ssize_t>(n_basis + 1)) {
    throw std::runtime_error("col_ptr length must be n_basis+1");
  }

  const int nnz = static_cast<int>(ham_val.shape(0));
  if (row_ind_1based.shape(0) != static_cast<py::ssize_t>(nnz)) {
    throw std::runtime_error("row_ind length must equal nnz");
  }

  int world_size_csc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_csc);
  const bool force_single_proc_csc =
      world_size_csc > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));
  const MPI_Comm csc_comm = force_single_proc_csc ? MPI_COMM_SELF : MPI_COMM_WORLD;

  // ELSI handle with sparse CSC format: matrix_format=1 (PEXSI_CSC)
  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 1, n_basis, n_electron, n_state);

  c_elsi_set_mpi(h, MPI_Comm_c2f(csc_comm));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt_csc = blacs_ctxt_1x1(csc_comm);
  c_elsi_set_blacs(h, ictxt_csc, 32);

  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  // Register sparsity pattern (single rank: nnz_l=nnz, n_lcol=n_basis)
  c_elsi_set_csc(h, nnz, nnz, n_basis, row_ind_1based.mutable_data(), col_ptr_1based.mutable_data());

  // Overlap values
  py::array_t<double, py::array::c_style | py::array::forcecast> ovlp_val;
  py::array_t<double, py::array::c_style> ovlp_dummy;
  double* ovlp_ptr = nullptr;

  if (ovlp_val_obj.is_none()) {
    ovlp_dummy = py::array_t<double, py::array::c_style>({nnz});
    std::memset(ovlp_dummy.mutable_data(), 0, static_cast<std::size_t>(nnz) * sizeof(double));
    ovlp_ptr = ovlp_dummy.mutable_data();
    c_elsi_set_unit_ovlp(h, 1);
  } else {
    ovlp_val = ovlp_val_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (ovlp_val.ndim() != 1 || ovlp_val.shape(0) != nnz) throw std::runtime_error("ovlp_val must be 1D length nnz");
    ovlp_ptr = ovlp_val.mutable_data();
    c_elsi_set_unit_ovlp(h, 0);
  }

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    auto k = py::reinterpret_borrow<py::str>(item.first);
    auto v = py::reinterpret_borrow<py::object>(item.second);
    opts[k.cast<std::string>()] = std::move(v);
  }
  apply_options(h, opts);

  py::array_t<double, py::array::c_style> dm_val({nnz});
  py::array_t<double> energy({1});
  c_elsi_dm_real_sparse(h, ham_val.mutable_data(), ovlp_ptr, dm_val.mutable_data(), energy.mutable_data());
  c_elsi_finalize(h);

  return py::make_tuple(dm_val, energy.at(0));
}

static py::tuple elsi_dm_real_coo(py::array_t<double, py::array::c_style | py::array::forcecast> ham_val,
                                  py::array_t<int, py::array::c_style | py::array::forcecast> row_ind_1based,
                                  py::array_t<int, py::array::c_style | py::array::forcecast> col_ind_1based,
                                  int n_basis, py::object ovlp_val_obj, int solver, double n_electron, int n_state,
                                  py::dict backend_opts) {
  maybe_mpi_init();

  if (ham_val.ndim() != 1) throw std::runtime_error("ham_val must be 1D");
  if (row_ind_1based.ndim() != 1 || col_ind_1based.ndim() != 1) throw std::runtime_error("row/col must be 1D");
  const int nnz = static_cast<int>(ham_val.shape(0));
  if (row_ind_1based.shape(0) != nnz || col_ind_1based.shape(0) != nnz) throw std::runtime_error("row/col length must equal nnz");

  // matrix_format=3 (GENERIC_COO)
  int world_size_coo = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_coo);
  const bool force_single_proc_coo =
      world_size_coo > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));
  const MPI_Comm coo_comm = force_single_proc_coo ? MPI_COMM_SELF : MPI_COMM_WORLD;

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 3, n_basis, n_electron, n_state);

  c_elsi_set_mpi(h, MPI_Comm_c2f(coo_comm));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt = blacs_ctxt_1x1(coo_comm);
  c_elsi_set_blacs(h, ictxt, 32);

  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  c_elsi_set_coo(h, nnz, nnz, row_ind_1based.mutable_data(), col_ind_1based.mutable_data());

  py::array_t<double, py::array::c_style | py::array::forcecast> ovlp_val;
  py::array_t<double, py::array::c_style> ovlp_dummy;
  double* ovlp_ptr = nullptr;

  if (ovlp_val_obj.is_none()) {
    ovlp_dummy = py::array_t<double, py::array::c_style>({nnz});
    std::memset(ovlp_dummy.mutable_data(), 0, static_cast<std::size_t>(nnz) * sizeof(double));
    ovlp_ptr = ovlp_dummy.mutable_data();
    c_elsi_set_unit_ovlp(h, 1);
  } else {
    ovlp_val = ovlp_val_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (ovlp_val.ndim() != 1 || ovlp_val.shape(0) != nnz) throw std::runtime_error("ovlp_val must be 1D length nnz");
    ovlp_ptr = ovlp_val.mutable_data();
    c_elsi_set_unit_ovlp(h, 0);
  }

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    auto k = py::reinterpret_borrow<py::str>(item.first);
    auto v = py::reinterpret_borrow<py::object>(item.second);
    opts[k.cast<std::string>()] = std::move(v);
  }
  apply_options(h, opts);

  py::array_t<double, py::array::c_style> dm_val({nnz});
  py::array_t<double> energy({1});
  c_elsi_dm_real_sparse(h, ham_val.mutable_data(), ovlp_ptr, dm_val.mutable_data(), energy.mutable_data());
  c_elsi_finalize(h);

  return py::make_tuple(dm_val, energy.at(0));
}

static py::tuple elsi_ev_complex_dense(
    py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast> H,
    py::object S_obj, int solver, double n_electron, int n_state, py::dict backend_opts, bool want_vectors) {
  maybe_mpi_init();

  if (H.ndim() != 2 || H.shape(0) != H.shape(1)) throw std::runtime_error("H must be a square 2D array");
  const int n_basis = static_cast<int>(H.shape(0));

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast> S;
  py::array_t<std::complex<double>, py::array::f_style> S_dummy;
  void* ovlp = nullptr;

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
  const MPI_Fint comm_f = MPI_Comm_c2f(force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD);
  c_elsi_set_mpi(h, comm_f);
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt = blacs_ctxt_1x1(force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD);
  c_elsi_set_blacs(h, ictxt, 32);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  if (!S_obj.is_none()) {
    S = S_obj.cast<py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast>>();
    if (S.ndim() != 2 || S.shape(0) != n_basis || S.shape(1) != n_basis) throw std::runtime_error("S must be (n,n)");
    ovlp = S.mutable_data();
    c_elsi_set_unit_ovlp(h, 0);
  } else {
    S_dummy = py::array_t<std::complex<double>, py::array::f_style>({n_basis, n_basis});
    auto* p = static_cast<std::complex<double>*>(S_dummy.mutable_data());
    std::fill(p, p + static_cast<std::size_t>(n_basis) * static_cast<std::size_t>(n_basis), std::complex<double>(0.0, 0.0));
    for (int j = 0; j < n_basis; ++j) {
      p[j * n_basis + j] = std::complex<double>(1.0, 0.0);
    }
    ovlp = S_dummy.mutable_data();
    c_elsi_set_unit_ovlp(h, 1);
  }

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    auto k = py::reinterpret_borrow<py::str>(item.first);
    auto v = py::reinterpret_borrow<py::object>(item.second);
    opts[k.cast<std::string>()] = std::move(v);
  }
  apply_options(h, opts);

  py::array_t<double> eval({n_basis});
  py::array_t<std::complex<double>, py::array::f_style> evec({n_basis, n_basis});

  void* ham = H.mutable_data();
  double* ev = eval.mutable_data();
  void* vec = want_vectors ? evec.mutable_data() : nullptr;

  if (want_vectors) {
    c_elsi_ev_complex(h, reinterpret_cast<double _Complex*>(ham), reinterpret_cast<double _Complex*>(ovlp), ev,
                      reinterpret_cast<double _Complex*>(vec));
  } else {
    c_elsi_ev_complex(h, reinterpret_cast<double _Complex*>(ham), reinterpret_cast<double _Complex*>(ovlp), ev,
                      reinterpret_cast<double _Complex*>(evec.mutable_data()));
  }

  c_elsi_finalize(h);
  if (!want_vectors) return py::make_tuple(eval, py::none());
  return py::make_tuple(eval, evec);
}

static py::tuple elsi_dm_complex_dense(
    py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast> H,
    py::object S_obj, int solver, double n_electron, int n_state, py::dict backend_opts) {
  maybe_mpi_init();
  if (H.ndim() != 2 || H.shape(0) != H.shape(1)) throw std::runtime_error("H must be a square 2D array");
  const int n_basis = static_cast<int>(H.shape(0));

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast> S;
  py::array_t<std::complex<double>, py::array::f_style> S_dummy;
  void* ovlp = nullptr;

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
  const MPI_Fint comm_f = MPI_Comm_c2f(force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD);
  c_elsi_set_mpi(h, comm_f);
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt = blacs_ctxt_1x1(force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD);
  c_elsi_set_blacs(h, ictxt, 32);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  if (!S_obj.is_none()) {
    S = S_obj.cast<py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast>>();
    if (S.ndim() != 2 || S.shape(0) != n_basis || S.shape(1) != n_basis) throw std::runtime_error("S must be (n,n)");
    ovlp = S.mutable_data();
    c_elsi_set_unit_ovlp(h, 0);
  } else {
    S_dummy = py::array_t<std::complex<double>, py::array::f_style>({n_basis, n_basis});
    auto* p = static_cast<std::complex<double>*>(S_dummy.mutable_data());
    std::fill(p, p + static_cast<std::size_t>(n_basis) * static_cast<std::size_t>(n_basis), std::complex<double>(0.0, 0.0));
    for (int j = 0; j < n_basis; ++j) {
      p[j * n_basis + j] = std::complex<double>(1.0, 0.0);
    }
    ovlp = S_dummy.mutable_data();
    c_elsi_set_unit_ovlp(h, 1);
  }

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    auto k = py::reinterpret_borrow<py::str>(item.first);
    auto v = py::reinterpret_borrow<py::object>(item.second);
    opts[k.cast<std::string>()] = std::move(v);
  }
  apply_options(h, opts);

  py::array_t<std::complex<double>, py::array::f_style> dm({n_basis, n_basis});
  py::array_t<double> energy({1});

  c_elsi_dm_complex(h, reinterpret_cast<double _Complex*>(H.mutable_data()), reinterpret_cast<double _Complex*>(ovlp),
                    reinterpret_cast<double _Complex*>(dm.mutable_data()), energy.mutable_data());
  c_elsi_finalize(h);
  return py::make_tuple(dm, energy.at(0));
}

PYBIND11_MODULE(_pyelsi_core, m) {
  m.doc() = "pyELSI compiled core";
  m.def("build_info", &build_info_dict);
  m.def("elsi_ev_real_dense", &elsi_ev_real_dense, py::arg("H"), py::arg("S"), py::arg("solver"), py::arg("n_electron"),
        py::arg("n_state"), py::arg("backend_opts"), py::arg("want_vectors"));
  m.def("elsi_dm_real_dense", &elsi_dm_real_dense, py::arg("H"), py::arg("S"), py::arg("solver"), py::arg("n_electron"),
        py::arg("n_state"), py::arg("backend_opts"));
  m.def("elsi_ev_complex_dense", &elsi_ev_complex_dense, py::arg("H"), py::arg("S"), py::arg("solver"), py::arg("n_electron"),
        py::arg("n_state"), py::arg("backend_opts"), py::arg("want_vectors"));
  m.def("elsi_dm_complex_dense", &elsi_dm_complex_dense, py::arg("H"), py::arg("S"), py::arg("solver"), py::arg("n_electron"),
        py::arg("n_state"), py::arg("backend_opts"));
  m.def("elsi_dm_real_csc", &elsi_dm_real_csc, py::arg("ham_val"), py::arg("row_ind_1based"), py::arg("col_ptr_1based"),
        py::arg("n_basis"), py::arg("ovlp_val"), py::arg("solver"), py::arg("n_electron"), py::arg("n_state"),
        py::arg("backend_opts"));
  m.def("elsi_dm_real_coo", &elsi_dm_real_coo, py::arg("ham_val"), py::arg("row_ind_1based"), py::arg("col_ind_1based"),
        py::arg("n_basis"), py::arg("ovlp_val"), py::arg("solver"), py::arg("n_electron"), py::arg("n_state"),
        py::arg("backend_opts"));
}

