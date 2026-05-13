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
int  Csys2blacs_handle(MPI_Comm comm);
void Cblacs_get(int context, int request, int* value);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);
// ScaLAPACK: compute the number of rows/columns owned by a given process
int  numroc_(const int* n, const int* nb, const int* iproc,
             const int* isrcproc, const int* nprocs);
// ScaLAPACK: initialise a distributed-array descriptor
void descinit_(int* desc, const int* m, const int* n,
               const int* mb, const int* nb,
               const int* irsrc, const int* icsrc,
               const int* ictxt, const int* lld, int* info);
}

static py::dict build_info_dict() {
  py::dict d;
  d["has_mpi"] = true;

#if PYELSI_HAS_CUDA
  d["has_cuda"] = true;
#else
  d["has_cuda"] = false;
#endif

#if PYELSI_HAS_CHASE
  d["has_chase"] = true;
#else
  d["has_chase"] = false;
#endif

#if PYELSI_HAS_SIPS
  d["has_sips"] = true;
#else
  d["has_sips"] = false;
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

// Return the most square (nprow, npcol) factorisation of `nprocs`.
static std::pair<int,int> best_pgrid(int nprocs) {
  int nprow = static_cast<int>(std::sqrt(static_cast<double>(nprocs)));
  while (nprocs % nprow != 0) --nprow;
  return {nprow, nprocs / nprow};
}

// Create a BLACS context for `comm` using the most square-ish process grid.
// For MPI_COMM_SELF (1 rank) this is 1×1; for MPI_COMM_WORLD with P ranks
// it is the factorisation closest to √P × √P.  All ranks in `comm` receive a
// valid context (unlike the old hardcoded 1×1 which left ranks 1..P-1 with
// context = -1 when called with MPI_COMM_WORLD).
static int blacs_ctxt_1x1(MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  auto [nprow, npcol] = best_pgrid(size);
  int ictxt = Csys2blacs_handle(comm);
  Cblacs_gridinit(&ictxt, "r", nprow, npcol);
  return ictxt;
}

// Copy local block-cyclic portion out of a global Fortran-col-major matrix.
// All arguments use 0-based local and global indices.
static void extract_block_cyclic(
    const double* global, double* local,
    int n, int nb, int nprow, int npcol,
    int myrow, int mycol, int local_rows, int local_cols)
{
  for (int jloc = 0; jloc < local_cols; ++jloc) {
    const int gj = ((jloc / nb) * npcol + mycol) * nb + (jloc % nb);
    for (int iloc = 0; iloc < local_rows; ++iloc) {
      const int gi = ((iloc / nb) * nprow + myrow) * nb + (iloc % nb);
      // Fortran col-major: element (gi, gj) lives at global[gi + gj*n]
      local[iloc + jloc * local_rows] =
          (gi < n && gj < n) ? global[gi + (std::size_t)gj * n] : 0.0;
    }
  }
}

// Scatter local block-cyclic data back into a zero-initialised global buffer.
// After calling this on all ranks, MPI_Allreduce(SUM) reconstructs the full matrix.
static void collect_block_cyclic(
    const double* local, double* global,
    int n, int nb, int nprow, int npcol,
    int myrow, int mycol, int local_rows, int local_cols)
{
  for (int jloc = 0; jloc < local_cols; ++jloc) {
    const int gj = ((jloc / nb) * npcol + mycol) * nb + (jloc % nb);
    for (int iloc = 0; iloc < local_rows; ++iloc) {
      const int gi = ((iloc / nb) * nprow + myrow) * nb + (iloc % nb);
      if (gi < n && gj < n)
        global[gi + (std::size_t)gj * n] = local[iloc + jloc * local_rows];
    }
  }
}

// Build the ScaLAPACK descriptor for a distributed n×n matrix.
// Sets lld = max(1, local_rows) and throws on descinit_ error.
static void make_dist_desc(int* desc, int n_basis, int nb, int ictxt,
                            int local_rows)
{
  const int izero = 0;
  int lld = std::max(1, local_rows);
  int info = 0;
  descinit_(desc, &n_basis, &n_basis, &nb, &nb,
            &izero, &izero, &ictxt, &lld, &info);
  if (info != 0)
    throw std::runtime_error(
        "descinit_ failed (info=" + std::to_string(info) + ")");
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

  // SIPS (SLEPc-SIP) settings
  if (auto v = get_int("sips_n_elpa"))         c_elsi_set_sips_n_elpa(h, *v);
  if (auto v = get_int("sips_n_slice"))         c_elsi_set_sips_n_slice(h, *v);
  if (auto v = get_double("sips_inertia_tol"))  c_elsi_set_sips_inertia_tol(h, *v);
  if (auto v = get_double("sips_ev_min"))       c_elsi_set_sips_ev_min(h, *v);
  if (auto v = get_double("sips_ev_max"))       c_elsi_set_sips_ev_max(h, *v);

  // ChASE settings
  if (auto v = get_double("chase_tol"))           c_elsi_set_chase_tol(h, *v);
  if (auto v = get_int("chase_filter_deg"))        c_elsi_set_chase_filter_deg(h, *v);
  if (auto v = get_double("chase_extra_space"))    c_elsi_set_chase_extra_space(h, *v);
  if (auto v = get_int("chase_min_extra_space"))   c_elsi_set_chase_min_extra_space(h, *v);
  if (auto v = get_int("chase_cholqr"))            c_elsi_set_chase_cholqr(h, *v);
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

  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  // Build opts map once
  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    opts[py::reinterpret_borrow<py::str>(item.first).cast<std::string>()] =
        py::reinterpret_borrow<py::object>(item.second);
  }

  // Overlap matrix (global, on all ranks)
  py::array_t<double, py::array::f_style | py::array::forcecast> S;
  py::array_t<double, py::array::f_style> S_dummy;
  double* ovlp_global = nullptr;
  const bool unit_ovlp = S_obj.is_none();
  if (!S_obj.is_none()) {
    S = S_obj.cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    if (S.ndim() != 2 || S.shape(0) != n_basis || S.shape(1) != n_basis)
      throw std::runtime_error("S must be (n,n)");
    ovlp_global = static_cast<double*>(S.mutable_data());
  } else {
    S_dummy = identity_fortran_colmajor(n_basis);
    ovlp_global = static_cast<double*>(S_dummy.mutable_data());
  }

  // ---- Serial / force_single_proc path ----
  if (force_single_proc || world_size == 1) {
    elsi_handle h = nullptr;
    c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
    const MPI_Comm comm = force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD;
    c_elsi_set_mpi(h, MPI_Comm_c2f(comm));
    c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
    c_elsi_set_blacs(h, blacs_ctxt_1x1(comm), 32);
    c_elsi_set_spin_degeneracy(h, 1.0);
    c_elsi_set_mu_broaden_scheme(h, 0);
    c_elsi_set_mu_broaden_width(h, 1.0e-12);
    c_elsi_set_unit_ovlp(h, unit_ovlp ? 1 : 0);
    apply_options(h, opts);

    py::array_t<double> eval({n_basis});
    py::array_t<double, py::array::f_style> evec({n_basis, n_basis});
    double* ham = static_cast<double*>(H.mutable_data());
    c_elsi_ev_real(h, ham, ovlp_global,
                   static_cast<double*>(eval.mutable_data()),
                   static_cast<double*>(evec.mutable_data()));
    c_elsi_finalize(h);
    if (!want_vectors) return py::make_tuple(eval, py::none());
    return py::make_tuple(eval, evec);
  }

  // ---- Distributed MPI path ----
  auto [nprow, npcol] = best_pgrid(world_size);
  const int nb = 32;

  int ictxt = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&ictxt, "r", nprow, npcol);
  int nprow_q, npcol_q, myrow, mycol;
  Cblacs_gridinfo(ictxt, &nprow_q, &npcol_q, &myrow, &mycol);

  const int izero = 0;
  int local_rows = numroc_(&n_basis, &nb, &myrow, &izero, &nprow_q);
  int local_cols = numroc_(&n_basis, &nb, &mycol, &izero, &npcol_q);

  int desc[9]; make_dist_desc(desc, n_basis, nb, ictxt, local_rows);

  const std::size_t local_sz = static_cast<std::size_t>(std::max(1, local_rows * local_cols));
  std::vector<double> H_local(local_sz, 0.0);
  std::vector<double> S_local(local_sz, 0.0);
  std::vector<double> evec_local(local_sz, 0.0);

  extract_block_cyclic(static_cast<const double*>(H.data()), H_local.data(),
                       n_basis, nb, nprow_q, npcol_q, myrow, mycol,
                       local_rows, local_cols);
  if (!unit_ovlp)
    extract_block_cyclic(ovlp_global, S_local.data(),
                         n_basis, nb, nprow_q, npcol_q, myrow, mycol,
                         local_rows, local_cols);

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
  c_elsi_set_mpi(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  c_elsi_set_blacs(h, ictxt, nb);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);
  c_elsi_set_unit_ovlp(h, unit_ovlp ? 1 : 0);
  apply_options(h, opts);

  py::array_t<double> eval({n_basis});
  c_elsi_ev_real(h, H_local.data(),
                 unit_ovlp ? S_local.data() : S_local.data(),
                 static_cast<double*>(eval.mutable_data()),
                 evec_local.data());
  c_elsi_finalize(h);

  // Eigenvalues are replicated on all ranks by ELPA — broadcast from rank 0
  // to be safe with other solvers.
  MPI_Bcast(static_cast<double*>(eval.mutable_data()), n_basis, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

  py::array_t<double, py::array::f_style> evec({n_basis, n_basis});
  if (want_vectors) {
    double* evec_ptr = static_cast<double*>(evec.mutable_data());
    std::memset(evec_ptr, 0, static_cast<std::size_t>(n_basis) * n_basis * sizeof(double));
    collect_block_cyclic(evec_local.data(), evec_ptr,
                         n_basis, nb, nprow_q, npcol_q, myrow, mycol,
                         local_rows, local_cols);
    MPI_Allreduce(MPI_IN_PLACE, evec_ptr,
                  static_cast<int>(static_cast<std::size_t>(n_basis) * n_basis),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  Cblacs_gridexit(ictxt);

  if (!want_vectors) return py::make_tuple(eval, py::none());
  return py::make_tuple(eval, evec);
}

static py::tuple elsi_dm_real_dense(py::array_t<double, py::array::f_style | py::array::forcecast> H,
                                    py::object S_obj, int solver, double n_electron, int n_state,
                                    py::dict backend_opts) {
  maybe_mpi_init();
  if (H.ndim() != 2 || H.shape(0) != H.shape(1)) throw std::runtime_error("H must be a square 2D array");
  const int n_basis = static_cast<int>(H.shape(0));

  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    opts[py::reinterpret_borrow<py::str>(item.first).cast<std::string>()] =
        py::reinterpret_borrow<py::object>(item.second);
  }

  py::array_t<double, py::array::f_style | py::array::forcecast> S;
  py::array_t<double, py::array::f_style> S_dummy;
  double* ovlp_global = nullptr;
  const bool unit_ovlp = S_obj.is_none();
  if (!S_obj.is_none()) {
    S = S_obj.cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    if (S.ndim() != 2 || S.shape(0) != n_basis || S.shape(1) != n_basis)
      throw std::runtime_error("S must be (n,n)");
    ovlp_global = static_cast<double*>(S.mutable_data());
  } else {
    S_dummy = identity_fortran_colmajor(n_basis);
    ovlp_global = static_cast<double*>(S_dummy.mutable_data());
  }

  // ---- Serial / force_single_proc path ----
  if (force_single_proc || world_size == 1) {
    elsi_handle h = nullptr;
    c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
    const MPI_Comm comm = force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD;
    c_elsi_set_mpi(h, MPI_Comm_c2f(comm));
    c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
    c_elsi_set_blacs(h, blacs_ctxt_1x1(comm), 32);
    c_elsi_set_spin_degeneracy(h, 1.0);
    c_elsi_set_mu_broaden_scheme(h, 0);
    c_elsi_set_mu_broaden_width(h, 1.0e-12);
    c_elsi_set_unit_ovlp(h, unit_ovlp ? 1 : 0);
    apply_options(h, opts);

    py::array_t<double, py::array::f_style> dm({n_basis, n_basis});
    double energy_val = 0.0;
    c_elsi_dm_real(h, static_cast<double*>(H.mutable_data()), ovlp_global,
                   static_cast<double*>(dm.mutable_data()), &energy_val);
    c_elsi_finalize(h);
    return py::make_tuple(dm, energy_val);
  }

  // ---- Distributed MPI path ----
  auto [nprow, npcol] = best_pgrid(world_size);
  const int nb = 32;

  int ictxt = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&ictxt, "r", nprow, npcol);
  int nprow_q, npcol_q, myrow, mycol;
  Cblacs_gridinfo(ictxt, &nprow_q, &npcol_q, &myrow, &mycol);

  const int izero = 0;
  int local_rows = numroc_(&n_basis, &nb, &myrow, &izero, &nprow_q);
  int local_cols = numroc_(&n_basis, &nb, &mycol, &izero, &npcol_q);

  int desc[9]; make_dist_desc(desc, n_basis, nb, ictxt, local_rows);

  const std::size_t local_sz = static_cast<std::size_t>(std::max(1, local_rows * local_cols));
  std::vector<double> H_local(local_sz, 0.0);
  std::vector<double> S_local(local_sz, 0.0);
  std::vector<double> DM_local(local_sz, 0.0);

  extract_block_cyclic(static_cast<const double*>(H.data()), H_local.data(),
                       n_basis, nb, nprow_q, npcol_q, myrow, mycol,
                       local_rows, local_cols);
  if (!unit_ovlp)
    extract_block_cyclic(ovlp_global, S_local.data(),
                         n_basis, nb, nprow_q, npcol_q, myrow, mycol,
                         local_rows, local_cols);

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 0, n_basis, n_electron, n_state);
  c_elsi_set_mpi(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  c_elsi_set_blacs(h, ictxt, nb);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);
  c_elsi_set_unit_ovlp(h, unit_ovlp ? 1 : 0);
  apply_options(h, opts);

  double energy_val = 0.0;
  c_elsi_dm_real(h, H_local.data(), S_local.data(), DM_local.data(), &energy_val);
  c_elsi_finalize(h);

  // Gather DM: each rank fills its slice of a zero matrix, then sum-reduce
  py::array_t<double, py::array::f_style> dm({n_basis, n_basis});
  double* dm_ptr = static_cast<double*>(dm.mutable_data());
  std::memset(dm_ptr, 0, static_cast<std::size_t>(n_basis) * n_basis * sizeof(double));
  collect_block_cyclic(DM_local.data(), dm_ptr,
                       n_basis, nb, nprow_q, npcol_q, myrow, mycol,
                       local_rows, local_cols);
  MPI_Allreduce(MPI_IN_PLACE, dm_ptr,
                static_cast<int>(static_cast<std::size_t>(n_basis) * n_basis),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(&energy_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  Cblacs_gridexit(ictxt);

  return py::make_tuple(dm, energy_val);
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
  if (row_ind_1based.shape(0) != nnz || col_ind_1based.shape(0) != nnz)
    throw std::runtime_error("row/col length must equal nnz");

  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const bool force_single_proc =
      world_size > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") && backend_opts["force_single_proc"].cast<int>() != 0));

  std::map<std::string, py::object> opts;
  for (auto item : backend_opts) {
    opts[py::reinterpret_borrow<py::str>(item.first).cast<std::string>()] =
        py::reinterpret_borrow<py::object>(item.second);
  }

  // Overlap: load once (needed for both serial and distributed paths)
  py::array_t<double, py::array::c_style | py::array::forcecast> ovlp_val;
  py::array_t<double, py::array::c_style> ovlp_dummy;
  const double* ovlp_global = nullptr;
  const bool unit_ovlp = ovlp_val_obj.is_none();
  if (!ovlp_val_obj.is_none()) {
    ovlp_val = ovlp_val_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (ovlp_val.ndim() != 1 || ovlp_val.shape(0) != nnz)
      throw std::runtime_error("ovlp_val must be 1D length nnz");
    ovlp_global = ovlp_val.data();
  } else {
    ovlp_dummy = py::array_t<double, py::array::c_style>({nnz});
    std::memset(ovlp_dummy.mutable_data(), 0, static_cast<std::size_t>(nnz) * sizeof(double));
    ovlp_global = ovlp_dummy.data();
  }

  const double* ham_data = ham_val.data();
  const int*    row_data = row_ind_1based.data();
  const int*    col_data = col_ind_1based.data();

  // ---- Serial / force_single_proc path ----
  if (force_single_proc || world_size == 1) {
    const MPI_Comm comm = force_single_proc ? MPI_COMM_SELF : MPI_COMM_WORLD;
    elsi_handle h = nullptr;
    c_elsi_init(&h, solver, 1, 3, n_basis, n_electron, n_state);
    c_elsi_set_mpi(h, MPI_Comm_c2f(comm));
    c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
    c_elsi_set_blacs(h, blacs_ctxt_1x1(comm), 32);
    c_elsi_set_spin_degeneracy(h, 1.0);
    c_elsi_set_mu_broaden_scheme(h, 0);
    c_elsi_set_mu_broaden_width(h, 1.0e-12);
    c_elsi_set_unit_ovlp(h, unit_ovlp ? 1 : 0);
    c_elsi_set_coo(h, nnz, nnz,
                   row_ind_1based.mutable_data(), col_ind_1based.mutable_data());
    apply_options(h, opts);

    py::array_t<double, py::array::c_style> dm_val_out({nnz});
    double energy_val = 0.0;
    c_elsi_dm_real_sparse(h, const_cast<double*>(ham_data),
                          const_cast<double*>(ovlp_global),
                          dm_val_out.mutable_data(), &energy_val);
    c_elsi_finalize(h);
    return py::make_tuple(dm_val_out, energy_val);
  }

  // ---- Distributed MPI path ----
  // Each rank owns rows [row_start, row_end).  We extract the local non-zeros
  // (those whose row index falls in the local range) and pass only those to
  // ELSI.  After the solve the local DM values are allreduced back to every
  // rank in the original nnz-length ordering.
  const int row_start = (world_rank * n_basis) / world_size;
  const int row_end   = ((world_rank + 1) * n_basis) / world_size;

  std::vector<double> H_loc, S_loc;
  std::vector<int>    R_loc, C_loc;
  std::vector<int>    glob_idx;  // position in the global [0, nnz) array

  for (int i = 0; i < nnz; ++i) {
    const int gi = row_data[i] - 1;  // 0-based global row
    if (gi >= row_start && gi < row_end) {
      H_loc.push_back(ham_data[i]);
      S_loc.push_back(ovlp_global[i]);
      R_loc.push_back(row_data[i]);
      C_loc.push_back(col_data[i]);
      glob_idx.push_back(i);
    }
  }
  const int local_nnz = static_cast<int>(H_loc.size());

  // Set up a proper process grid so all ranks get a valid BLACS context
  auto [nprow, npcol] = best_pgrid(world_size);
  int ictxt = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&ictxt, "r", nprow, npcol);

  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 3, n_basis, n_electron, n_state);
  c_elsi_set_mpi(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  c_elsi_set_blacs(h, ictxt, 32);
  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);
  c_elsi_set_unit_ovlp(h, unit_ovlp ? 1 : 0);

  // global_nnz = nnz (each non-zero belongs to exactly one rank)
  c_elsi_set_coo(h, nnz, local_nnz, R_loc.data(), C_loc.data());
  apply_options(h, opts);

  std::vector<double> DM_loc(std::max(1, local_nnz), 0.0);
  double energy_val = 0.0;
  c_elsi_dm_real_sparse(h,
                        H_loc.empty() ? nullptr : H_loc.data(),
                        S_loc.empty() ? nullptr : S_loc.data(),
                        DM_loc.data(), &energy_val);
  c_elsi_finalize(h);
  Cblacs_gridexit(ictxt);

  // Gather: each rank fills its slice of a full zero array, then Allreduce(SUM)
  py::array_t<double, py::array::c_style> dm_val_out({nnz});
  double* dm_ptr = dm_val_out.mutable_data();
  std::memset(dm_ptr, 0, static_cast<std::size_t>(nnz) * sizeof(double));
  for (int k = 0; k < local_nnz; ++k)
    dm_ptr[glob_idx[k]] = DM_loc[k];
  MPI_Allreduce(MPI_IN_PLACE, dm_ptr, nnz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(&energy_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return py::make_tuple(dm_val_out, energy_val);
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

/** Sparse (GENERIC_COO) eigenproblem solver — used by SIPS (SLEPc-SIP).
 *
 * Computes the n_state lowest eigenvalues/eigenvectors of a sparse symmetric
 * matrix H (supplied as 1-based COO triplets).  Eigenvectors are returned in
 * BLACS column-major layout; for a 1×1 grid the array has shape
 * (n_basis, n_basis) with the first n_state columns populated.
 */
static py::tuple elsi_ev_real_coo(
    py::array_t<double, py::array::c_style | py::array::forcecast> ham_val,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ind_1based,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_ind_1based,
    int n_basis, py::object ovlp_val_obj, int solver,
    double n_electron, int n_state,
    py::dict backend_opts, bool want_vectors) {

  maybe_mpi_init();

  if (ham_val.ndim() != 1) throw std::runtime_error("ham_val must be 1D");
  if (row_ind_1based.ndim() != 1 || col_ind_1based.ndim() != 1)
    throw std::runtime_error("row/col indices must be 1D");
  const int nnz = static_cast<int>(ham_val.shape(0));
  if (row_ind_1based.shape(0) != nnz || col_ind_1based.shape(0) != nnz)
    throw std::runtime_error("row/col length must equal nnz");

  int world_size_ev = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_ev);
  const bool force_single =
      world_size_ev > 1 &&
      ((std::getenv("PYELSI_FORCE_SINGLE_PROC") != nullptr) ||
       (backend_opts.contains("force_single_proc") &&
        backend_opts["force_single_proc"].cast<int>() != 0));
  const MPI_Comm ev_comm = force_single ? MPI_COMM_SELF : MPI_COMM_WORLD;

  // matrix_format=3 (GENERIC_COO), parallel_mode=1 (MULTI_PROC)
  elsi_handle h = nullptr;
  c_elsi_init(&h, solver, 1, 3, n_basis, n_electron, n_state);

  c_elsi_set_mpi(h, MPI_Comm_c2f(ev_comm));
  c_elsi_set_mpi_global(h, MPI_Comm_c2f(MPI_COMM_WORLD));
  const int ictxt_ev = blacs_ctxt_1x1(ev_comm);
  c_elsi_set_blacs(h, ictxt_ev, 32);

  c_elsi_set_spin_degeneracy(h, 1.0);
  c_elsi_set_mu_broaden_scheme(h, 0);
  c_elsi_set_mu_broaden_width(h, 1.0e-12);

  c_elsi_set_coo(h, nnz, nnz,
                 row_ind_1based.mutable_data(),
                 col_ind_1based.mutable_data());

  py::array_t<double, py::array::c_style | py::array::forcecast> ovlp_val;
  py::array_t<double, py::array::c_style> ovlp_dummy;
  double* ovlp_ptr = nullptr;

  if (ovlp_val_obj.is_none()) {
    ovlp_dummy = py::array_t<double, py::array::c_style>({nnz});
    std::memset(ovlp_dummy.mutable_data(), 0,
                static_cast<std::size_t>(nnz) * sizeof(double));
    ovlp_ptr = ovlp_dummy.mutable_data();
    c_elsi_set_unit_ovlp(h, 1);
  } else {
    ovlp_val = ovlp_val_obj.cast<
        py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (ovlp_val.ndim() != 1 || ovlp_val.shape(0) != nnz)
      throw std::runtime_error("ovlp_val must be 1D of length nnz");
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

  // Eigenvalues (n_state) and eigenvectors in BLACS layout (n_basis × n_basis).
  // For a 1×1 grid, n_lrow = n_lcol = n_basis; only the first n_state columns
  // of evec are populated by ELSI.
  py::array_t<double> eval({n_state});
  py::array_t<double, py::array::f_style> evec({n_basis, n_basis});

  double* ev  = static_cast<double*>(eval.mutable_data());
  double* vec = want_vectors
                  ? static_cast<double*>(evec.mutable_data())
                  : static_cast<double*>(evec.mutable_data()); // always alloc

  c_elsi_ev_real_sparse(h, ham_val.mutable_data(), ovlp_ptr, ev, vec);
  c_elsi_finalize(h);

  if (!want_vectors) return py::make_tuple(eval, py::none());
  return py::make_tuple(eval, evec);
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
  m.def("elsi_ev_real_coo", &elsi_ev_real_coo, py::arg("ham_val"), py::arg("row_ind_1based"), py::arg("col_ind_1based"),
        py::arg("n_basis"), py::arg("ovlp_val"), py::arg("solver"), py::arg("n_electron"), py::arg("n_state"),
        py::arg("backend_opts"), py::arg("want_vectors"));
}

