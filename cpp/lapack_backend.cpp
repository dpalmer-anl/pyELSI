#include "lapack_backend.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
// LAPACK Fortran symbols (OpenBLAS/Netlib naming). Most distributions provide these.
void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork,
            int* info);
void dsygv_(int* itype, char* jobz, char* uplo, int* n, double* a, int* lda, double* b, int* ldb, double* w,
            double* work, int* lwork, int* info);
}

namespace pyelsi {

static void check_info(int info, const char* fn) {
  if (info == 0) return;
  if (info < 0) throw std::runtime_error(std::string(fn) + ": illegal value in argument " + std::to_string(-info));
  throw std::runtime_error(std::string(fn) + ": algorithm failed to converge (info=" + std::to_string(info) + ")");
}

EighResult lapack_eigh(const double* H, const double* S, int n, bool generalized, bool want_vectors) {
  if (n <= 0) throw std::runtime_error("lapack_eigh: n must be > 0");
  if (!H) throw std::runtime_error("lapack_eigh: H is null");
  if (generalized && !S) throw std::runtime_error("lapack_eigh: generalized requires S");

  std::vector<double> a(static_cast<size_t>(n) * static_cast<size_t>(n));
  std::memcpy(a.data(), H, a.size() * sizeof(double));

  std::vector<double> b;
  if (generalized) {
    b.resize(a.size());
    std::memcpy(b.data(), S, b.size() * sizeof(double));
  }

  std::vector<double> w(static_cast<size_t>(n));

  char jobz = want_vectors ? 'V' : 'N';
  char uplo = 'U';
  int lda = n;
  int info = 0;

  // Workspace query
  int lwork = -1;
  double work_query = 0.0;

  if (!generalized) {
    dsyev_(&jobz, &uplo, &n, a.data(), &lda, w.data(), &work_query, &lwork, &info);
    check_info(info, "dsyev");
    lwork = static_cast<int>(work_query);
    std::vector<double> work(static_cast<size_t>(lwork));
    dsyev_(&jobz, &uplo, &n, a.data(), &lda, w.data(), work.data(), &lwork, &info);
    check_info(info, "dsyev");
  } else {
    int itype = 1;
    int ldb = n;
    dsygv_(&itype, &jobz, &uplo, &n, a.data(), &lda, b.data(), &ldb, w.data(), &work_query, &lwork, &info);
    check_info(info, "dsygv");
    lwork = static_cast<int>(work_query);
    std::vector<double> work(static_cast<size_t>(lwork));
    dsygv_(&itype, &jobz, &uplo, &n, a.data(), &lda, b.data(), &ldb, w.data(), work.data(), &lwork, &info);
    check_info(info, "dsygv");
  }

  EighResult r;
  r.w = std::move(w);
  r.n = n;
  r.has_vectors = want_vectors;
  if (want_vectors) {
    r.v = std::move(a);  // eigenvectors in A (column-major)
  }
  return r;
}

}  // namespace pyelsi

