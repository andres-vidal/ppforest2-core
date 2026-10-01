# Contributing to ppforest2-core

Thanks for your interest in contributing. This repository is the C++ engine,
its command-line tool and the benchmarks. The R package lives in
[ppforest2-r](https://github.com/andres-vidal/ppforest2-r), which vendors the
core from tagged releases of this repository.

## Reporting issues

- Search the [issue tracker](https://github.com/andres-vidal/ppforest2-core/issues)
  first to avoid duplicates.
- For bugs, please include a minimal reproducible example: the command or code,
  the data (or a small subset of it), the seed, and the output of
  `ppforest2 --version`.
- Issues with the R package, including its functions, plots and installation,
  belong in the
  [ppforest2-r issue tracker](https://github.com/andres-vidal/ppforest2-r/issues).

## Pull requests

- Open an issue describing the change before starting substantial work, so we
  can agree on the approach.
- Fork the repository and create a branch off `main`.
- Follow the existing code style and format C++ code with `clang-format`
  (`make format`).
- Add tests for any behaviour you change or add. Tests use GoogleTest and live
  next to the source they test (`Foo.test.cpp` next to `Foo.cpp`).
- Run the test suite with `make test` before submitting. For changes under
  `core/src/models/`, `stats/`, `serialization/` or `utils/`, which the R package
  compiles, also run `make cpp-strict`.
- Results must be identical across platforms for the same seed. Use
  `stats::RNG` for random numbers, `stats::Uniform::distinct()` for shuffling
  and `std::stable_sort` where the order of equal elements matters.
- If a change intentionally alters model output, regenerate the golden files
  with `make golden-regen` and commit them with the change. See
  [Reproducibility Break Protocol](README.md#reproducibility-break-protocol).
- Add an entry to `CHANGELOG.md` in the same commit as the change.

## Code of Conduct

By participating in this project you agree to abide by its
[Code of Conduct](CODE_OF_CONDUCT.md).
