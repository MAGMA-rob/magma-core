# magma-core
MAGMA core is one Python package with two dependency zones. Version 2.0.0b3
is the third beta of v2. This beta supports Python 3.12; simulation installation
was checked on Linux x86_64. Broader Python/platform support is not yet validated.

## Description

magma_core contains protocol, base classes and utils used by different magma packages.

## Installation

### Pip

```bash
pip install "magma_core==2.0.0b3"              # Simulator-independent interface
pip install "magma_core[interface]==2.0.0b3"   # Alias for the default install
pip install "magma_core[simulation]==2.0.0b3"  # Includes simulation dependencies
```

For a local checkout, use `pip install -e .` or `pip install -e ".[all]"`.
Extras add dependencies; they do not change which Python modules are shipped.
The default installation includes Torch, requests, PyYAML, Pydantic, and
`typing_extensions`, but does not require ManiSkill, SAPIEN, or Gymnasium.

### Github

```bash
pip install "git+https://github.com/MAGMA-rob/magma-core.git@main"
```

Install a pinned release tag:

```bash
pip install "magma_core[simulation] @ git+https://github.com/MAGMA-rob/magma-core.git@v2.0.0b3"
```

---

If you spot any documentation errors, problem in the code. Please contact me at `l.bernat@sileane.com`
