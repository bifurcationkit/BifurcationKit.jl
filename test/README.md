# Test organisation

## Directory tree

- Each test file must be placed in a subdirectory of  test .
- A test file must be a Julia file (ending with  .jl ).
- Each subdirectory is treated as a test category, and the files it contains are treated as separate tests.
- Consequently, the top-level  test  directory must not contain any test files.

Here is an example of the directory tree:

```
./
├── codim_2_po_collocation/
│   └── codim2PO-OColl.jl
├── codim_2_po_shooting/
│   └── codim2PO-shooting.jl
├── codim_2_po_shooting_mf/
│   └── codim2PO-shooting-mf.jl
├── condensation_of_parameters/
│   └── cop.jl
├── continuation/
│   ├── simple_continuation.jl
│   ├── test_bif_detection.jl
│   └── test-cont-non-vector.jl
...
├── results/
│   └── test_results.jl
├── runtests.jl
└── wave/
    └── test_wave.jl
```


## Main script: runtest.jl


`runtests.jl` is the launcher for all tests (or a subset of tests).

Its algorithm detects all subdirectories in the current directory (`test`) and finds all files contained within them.

The `runtests.jl` script accepts several arguments:

- `-a|--all`: run all tests
- `-n|--dry-run`: only print the names of the tests that will run (with the provided arguments)
- `filter_string`: a string specifying which tests to perform. The `filter_string` accepts file globbing patterns for tests (see examples)

## Examples on how to run tests locally

### Running all tests

```bash
julia -e 'Pkg.activate("."); Pkg.test()'
```

or

```bash
julia -e 'Pkg.activate("."); Pkg.test(test_args = [ "-a" ])'
```

### Run the tests contained in a specific directory (aka a specific category)

```bash
julia -e 'Pkg.activate("."); Pkg.test(test_args = [ "wave" ])'
```

or

```bash
julia -e 'Pkg.activate("."); Pkg.test(test_args = [ "wave/*" ])'
```

will run:

- `wave/test_wave.jl`

### Running several tests

```bash
julia -e 'Pkg.activate("."); Pkg.test(test_args = [ "periodic_orbits_*" ])'
```

will run:

- `periodic_orbits_bp_po/freire.jl`
- `periodic_orbits_function_fd/stuartLandauCollocation.jl`
- `periodic_orbits_function_fd/stuartLandauTrap.jl`
- `periodic_orbits_function_fd/test_potrap.jl`
- `periodic_orbits_function_sh1/test_SS.jl`
- `periodic_orbits_function_sh2/poincareMap.jl`
- `periodic_orbits_function_sh3/stuartLandauSH.jl`
- `periodic_orbits_function_sh4_and_collocation/testLure.jl`


### Verifying with tests will be run (whithout running them)

```bash
julia -e 'Pkg.activate("."); Pkg.test(test_args = [ "periodic_orbits_*", "-n" ])'
```

will display:

```
Activating project at `~/Development/BK/BifurcationKit.jl`
   Testing BifurcationKit
   Testing Running tests...
--> There are 1 threads
--> Dry run mode active. Tests will be listed but not executed.
[ Info: Dry run: would run periodic_orbits_bp_po/freire.jl
[ Info: Dry run: would run periodic_orbits_function_fd/stuartLandauCollocation.jl
[ Info: Dry run: would run periodic_orbits_function_fd/stuartLandauTrap.jl
[ Info: Dry run: would run periodic_orbits_function_fd/test_potrap.jl
[ Info: Dry run: would run periodic_orbits_function_sh1/test_SS.jl
[ Info: Dry run: would run periodic_orbits_function_sh2/poincareMap.jl
[ Info: Dry run: would run periodic_orbits_function_sh3/stuartLandauSH.jl
[ Info: Dry run: would run periodic_orbits_function_sh4_and_collocation/testLure.jl
Test Summary:  |Time
BifurcationKit | None  0.2s
     Testing BifurcationKit tests passed
```

## Passing arguments to runtests.jls on CI hosts using labels


The `ci.yml` file reads the **LABELS** associated with a GitHub pull request (PR) and uses the label names as `filter_string`s for the `runtests.jl` script.

For this to work, label names must follow this format:

- Start with the string: `Run test(s): ` (including the trailing space character)
- Followed by a `filter_string`, or multiple `filter_string`s separated by `|` (pipe character)

You can trigger multiple tests using either:
- A single label describing several tests, or
- Multiple labels, each providing one or more tests (any combination is valid)

**Examples of valid labels:**

- `Run test(s): wave`
- `Run test(s): wave/*`
- `Run test(s): periodic_orbits_*`
- `Run test(s): wave | newton`

**Examples of labels that will not trigger any tests:**

- `bug`
- `improvement`
- Any classic label already used in this project :)

**Examples of invalid labels (these will run all tests):**

- `Run test(s): ../examples*`
- `Run test(s): baddirectory`
