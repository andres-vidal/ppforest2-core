MAKEFLAGS += --no-print-directory

BUILD_DIR = .build
BUILD_DIR_DEBUG = .debug
BUILD_DIR_COV = .coverage
CORE_VERSION := $(shell cat VERSION)

NLHOMANN_JSON_HEADERS_PATH = ${BUILD_DIR}/_deps/json-src/include
PCG_HEADERS_PATH = ${BUILD_DIR}/_deps/pcg-src/include

CMAKE_EXTRA ?=

clean:
	@rm -rf ${BUILD_DIR} ${BUILD_DIR_DEBUG} ${BUILD_DIR_COV}

fetch-deps:
	@mkdir -p ${BUILD_DIR}
	@cd ${BUILD_DIR} && cmake -G "Unix Makefiles" \
		-DCMAKE_BUILD_TYPE=Release \
		-DPPFOREST2_CORE_ONLY=ON \
		${CMAKE_EXTRA} ../core

build:
	@mkdir -p ${BUILD_DIR}
	@cd ${BUILD_DIR} && cmake -G "Unix Makefiles" \
		-DCMAKE_BUILD_TYPE=Release -DPPFOREST2_CORE_ONLY=OFF \
		${CMAKE_EXTRA} ../core && make

build-debug:
	@mkdir -p ${BUILD_DIR_DEBUG}
	@cd ${BUILD_DIR_DEBUG} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} ../core && make

test: build
	@cd ./$(BUILD_DIR) && ./ppforest2-test

test-debug: build-debug
	@cd ./$(BUILD_DIR_DEBUG) && ./ppforest2-test

build-coverage:
	@mkdir -p ${BUILD_DIR_COV}
	@cd ${BUILD_DIR_COV} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DPPFOREST2_COVERAGE=ON ${CMAKE_EXTRA} ../core && make

test-coverage: build-coverage
	@cd ./$(BUILD_DIR_COV) && ./ppforest2-test

LCOV_IGNORE = --ignore-errors mismatch,inconsistent,unsupported,range,format,category,unused

coverage: test-coverage
	@lcov --capture --directory ${BUILD_DIR_COV} --output-file ${BUILD_DIR_COV}/coverage.info --quiet ${LCOV_IGNORE} --rc branch_coverage=0
	@lcov --extract ${BUILD_DIR_COV}/coverage.info '*/core/src/*' -o ${BUILD_DIR_COV}/coverage-filtered.info --quiet ${LCOV_IGNORE}
	@lcov --remove ${BUILD_DIR_COV}/coverage-filtered.info '*/_deps/*' '*.test.*' '*/golden/*' '*test.cpp' -o ${BUILD_DIR_COV}/coverage-filtered.info --quiet ${LCOV_IGNORE}
	@genhtml ${BUILD_DIR_COV}/coverage-filtered.info -o ${BUILD_DIR_COV}/html --quiet ${LCOV_IGNORE}
	@python3 scripts/coverage-report.py ${BUILD_DIR_COV}/html

golden-regen: build
	@cd ./$(BUILD_DIR) && ./ppforest2-golden-gen

# Dev tools (clang-format, clang-tidy, cppcheck via pip; doxygen via cmake)

TOOLS_DIR = .tools

install-tools:
	@pip install -r requirements-dev.txt
	@command -v asdf >/dev/null 2>&1 && asdf reshim python || true
	@mkdir -p ${TOOLS_DIR}
	@cd ${TOOLS_DIR} && cmake ../tools && make

install-doxygen:
	@mkdir -p ${TOOLS_DIR}
	@cd ${TOOLS_DIR} && cmake ../tools && make

clean-tools:
	@rm -rf ${TOOLS_DIR}

FORMAT_SOURCES = $(shell find core/src core/include -name '*.cpp' -o -name '*.hpp' -o -name '*.h')
TIDY_SOURCES = $(shell find core/src -name '*.cpp' ! -name '*.test.cpp')
# .hpp included too: `misc-include-cleaner` only reports at the top-level
# TU, so headers need to be scanned directly (mirroring clangd's per-file
# open behaviour).
INCLUDE_CHECK_SOURCES = $(shell find core/src -name '*.cpp' -o -name '*.hpp')
# Mirrors the `.clangd` IgnoreHeader list — third-party headers where the
# check misfires on by-value returns or ADL-resolved overloads.
INCLUDE_CHECK_IGNORE = nlohmann/json\.hpp;fmt/format\.h;serialization/Json\.hpp;serialization/JsonOptional\.hpp

PARALLEL = $$(nproc 2>/dev/null || sysctl -n hw.ncpu)

format:
	@clang-format -i ${FORMAT_SOURCES}

format-dry:
	@clang-format --dry-run --Werror ${FORMAT_SOURCES}

tidy: build
	@echo ${TIDY_SOURCES} | tr ' ' '\n' | xargs -P ${PARALLEL} -n 1 clang-tidy -p ${BUILD_DIR} --warnings-as-errors='*'
	@if echo ${INCLUDE_CHECK_SOURCES} | tr ' ' '\n' | \
		xargs -P ${PARALLEL} -n 1 \
		clang-tidy -p ${BUILD_DIR} --checks='-*,misc-include-cleaner' \
		--config="{CheckOptions: [{key: misc-include-cleaner.IgnoreHeaders, value: '${INCLUDE_CHECK_IGNORE}'}]}" \
		2>/dev/null | grep -E "not used directly"; then \
		echo "Unused includes found."; exit 1; \
	else \
		echo "No unused includes found."; \
	fi

# Static analysis over the sources the R package actually ships (the same scope
# as STRICT_SOURCES) plus the R glue. Three things keep --check-level=exhaustive
# from taking ~50 min:
#   - Scope: scanning the full core/src (cli/io/golden + GoogleTest-heavy
#     *.test.cpp, none of which ship) is ~170x the analysis cost. We scan only
#     the shipped core.
#   - --max-configs=1: the Rcpp glue headers expand into ~260 preprocessor
#     configurations; checking the default 12 under exhaustive cost ~4 min on
#     ppforest2.h alone. The package builds in one configuration, so we check one.
#   - --cppcheck-build-dir: caches per-header analysis so the large vendored
#     nlohmann/json header is analysed once, not re-analysed for every one of the
#     ~20 translation units that include it (~20x slower without it).
# Together the analysis finishes in seconds.
CPPCHECK_BUILD_DIR ?= .cppcheck-cache
analyze:
	@mkdir -p ${CPPCHECK_BUILD_DIR}
	@cppcheck --enable=all --check-level=exhaustive --inline-suppr --max-configs=1 \
		--cppcheck-build-dir=${CPPCHECK_BUILD_DIR} \
		--suppress=missingIncludeSystem --suppress=duplInheritedMember \
		--suppress=toomanyconfigs \
		--suppress='syntaxError:core/src/utils/UserError.hpp' \
		--quiet \
		${STRICT_SOURCES} \
		-Icore/src -Icore/include

# Strict-warning compile of the core sources the R package vendors, mirroring
# CRAN's stricter compilation. Uses real GCC (macOS `g++` is clang and won't
# catch these) — override on macOS with `make cpp-strict STRICT_CXX=g++-15`. Eigen
# comes from `.build/_deps` (run `make fetch-deps` first) or the system.
STRICT_CXX ?= g++
STRICT_SOURCES = $(shell find core/src -name '*.cpp' ! -name '*.test.cpp' ! -name 'test.cpp' ! -path '*/cli/*' ! -path '*/io/*' ! -path '*/golden/*')
cpp-strict:
	@eigen=$$(ls -d ${BUILD_DIR}/_deps/eigen-src 2>/dev/null || echo /usr/include/eigen3); \
	rc=0; \
	for f in ${STRICT_SOURCES}; do \
		${STRICT_CXX} -std=c++17 -O2 -fopenmp -Wall -Wextra -pedantic -Werror \
			-DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE -DEIGEN_NO_AUTOMATIC_RESIZING \
			-isystem $$eigen -isystem ${NLHOMANN_JSON_HEADERS_PATH} -isystem ${PCG_HEADERS_PATH} -Icore/src -Icore/include \
			-c $$f -o /dev/null || rc=1; \
	done; \
	[ $$rc = 0 ] && echo "OK: core compiles clean under -Wall -Wextra -pedantic -Werror" || exit 1

# Aggregate quality gates: format check, clang-tidy (incl. misc-include-cleaner),
# and cppcheck. No autofix — CI-friendly single entry point.
check: format-dry tidy analyze

# Documentation

DOCS_DIR = docs
DOCS_BUILD_DIR = ${DOCS_DIR}/.build
DOXYGEN = ${TOOLS_DIR}/doxygen/bin/doxygen
DOCS_REF ?= main

docs-site:
	@mkdir -p ${DOCS_BUILD_DIR}
	@sed 's/{{VERSION}}/v${CORE_VERSION}/g' ${DOCS_DIR}/index.html > ${DOCS_BUILD_DIR}/index.html
	@cp ${DOCS_DIR}/style.css ${DOCS_BUILD_DIR}/

docs-cpp:
	@mkdir -p ${DOCS_BUILD_DIR}/cpp
	@( cat ${DOCS_DIR}/Doxyfile ; echo "PROJECT_NUMBER = v${CORE_VERSION}" ) | ${DOXYGEN} -

docs: docs-site docs-cpp

# Release management

RELEASE_TAG ?= v${CORE_VERSION}

release:
	@git tag -a ${RELEASE_TAG} -m "Release ${RELEASE_TAG}"
	@git push origin ${RELEASE_TAG}

release-revert:
	@echo "This will delete the local and remote git tag '${RELEASE_TAG}'."
	@echo "Press Enter to continue or Ctrl-C to abort." && read _
	git tag -d ${RELEASE_TAG}
	git push origin :refs/tags/${RELEASE_TAG}
	@echo "Reverted release ${RELEASE_TAG}."
	@echo "Note: if a GitHub Release exists for this tag, delete it manually at"
	@echo "  https://github.com/$$(git remote get-url origin | sed 's|.*github.com[:/]||;s|\.git$$||')/releases"

# Benchmarking

# Benchmark scenarios split by mode so baselines track per-family and release
# notes can render them as separate tables. Both files must stay in defaults
# alignment (see the `_note` field at the top of each scenarios JSON).
BENCH_SCENARIOS_CLS = bench/default-scenarios-classification.json
BENCH_SCENARIOS_REG = bench/default-scenarios-regression.json
BENCH_REF ?= main

benchmark: build
	@echo "=== Classification benchmarks ==="
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_CLS}
	@echo ""
	@echo "=== Regression benchmarks (experimental) ==="
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_REG}

benchmark-save: build
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_CLS} -o bench/results-classification.json -o bench/results-classification.csv
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_REG} -o bench/results-regression.json -o bench/results-regression.csv

benchmark-compare: build
	@echo "=== Classification ==="
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_CLS} -b bench/results-classification.json
	@echo ""
	@echo "=== Regression (experimental) ==="
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_REG} -b bench/results-regression.json

benchmark-vs: build
	@echo "Building and benchmarking current branch..."
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_CLS} -o bench/.current-cls.json -q
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_REG} -o bench/.current-reg.json -q
	@echo "Setting up baseline (${BENCH_REF})..."
	@git worktree add -f .bench-worktree ${BENCH_REF} 2>/dev/null || { echo "Error: Could not create worktree for ref '${BENCH_REF}'"; exit 1; }
	@mkdir -p .bench-worktree/bench
	@cp ${BENCH_SCENARIOS_CLS} .bench-worktree/bench/default-scenarios-classification.json 2>/dev/null || true
	@cp ${BENCH_SCENARIOS_REG} .bench-worktree/bench/default-scenarios-regression.json 2>/dev/null || true
	@echo "Building baseline..."
	@cd .bench-worktree && mkdir -p .build && cd .build && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../core > /dev/null 2>&1 && make -j > /dev/null 2>&1
	@echo "Running baseline benchmarks..."
	@cd .bench-worktree && .build/ppforest2 benchmark -s bench/default-scenarios-classification.json -o bench/.baseline-cls.json -q
	@cd .bench-worktree && .build/ppforest2 benchmark -s bench/default-scenarios-regression.json -o bench/.baseline-reg.json -q
	@echo ""
	@echo "=== Classification ==="
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_CLS} -b .bench-worktree/bench/.baseline-cls.json
	@echo ""
	@echo "=== Regression (experimental) ==="
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS_REG} -b .bench-worktree/bench/.baseline-reg.json
	@rm -f bench/.current-cls.json bench/.current-reg.json
	@git worktree remove -f .bench-worktree 2>/dev/null || true