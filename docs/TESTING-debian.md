# Testing on Debian Trixie (Intel iGPU)

Second verification target for the OpenVINO device work (chant-omr#61):
a different Intel GPU (Alder Lake UHD) on an apt-based distro, as opposed
to the primary Fedora + Arc setup. This is a **correctness** data point,
not a benchmark — the UHD iGPU has far fewer EUs than Arc.

## 1. GPU driver stack

OpenVINO's GPU plugin needs NEO (OpenCL) + Level Zero:

```bash
sudo apt update && sudo apt install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero clinfo
sudo usermod -aG render "$USER"  # re-login afterwards
```

Debian is not on Intel's officially supported list (Ubuntu is), so those
packages may be absent or stale. Fallback: the manual `.deb`s from Intel's
[compute-runtime GitHub releases](https://github.com/intel/compute-runtime/releases)
— a known-working path on Debian 13.

Verify before anything else:

```bash
clinfo | grep -i "device name"  # iGPU must be listed
```

## 2. Environment

Trixie ships Python 3.13, which meets the `>=3.11` requirement, and
enforces PEP 668, so a venv is mandatory:

```bash
git clone https://github.com/pgquiles/chant-omr.git && cd chant-omr
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"
env -u LD_LIBRARY_PATH .venv/bin/python -c \
  "import openvino as ov; print(ov.Core().available_devices)"
```

`GPU` must appear in the device list. No XPU torch wheel is needed here:
the OV tests run on the plain pip torch build, which imports cleanly on
a OneAPI-free system.

## 3. What to run

```bash
env -u LD_LIBRARY_PATH .venv/bin/python -m pytest tests/test_export.py \
  -q --no-header -p no:cacheprovider
```

Expect all green, in particular
`TestOVDecodeIntegration::test_ov_auto_resolves_to_working_device`:
it loads all four IRs via `device="AUTO"` and runs greedy decode. With
the #61 fix, AUTO resolves to the explicit iGPU plugin; without it, the
shared AUTO context fails with `Cannot find tensor for port`.
