#pragma once

#include <optional>
#include <utility>

namespace pyelsi {

struct EighResult {
  // eigenvalues always returned
  // eigenvectors returned only when requested
  std::vector<double> w;
  std::vector<double> v;
  int n = 0;
  bool has_vectors = false;
};

EighResult lapack_eigh(const double* H, const double* S, int n, bool generalized, bool want_vectors);

}  // namespace pyelsi

