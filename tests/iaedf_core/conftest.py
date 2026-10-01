from pathlib import Path

import pytest

from memory_hole import iaedf_core

ASSETS = Path(iaedf_core.__file__).parent / "assets"


@pytest.fixture(scope="session")
def waveform_csv_text():
    return (ASSETS / "tailored_waveform_5harmonic.csv").read_text(encoding="utf-8-sig")


@pytest.fixture(scope="session")
def xsec_text():
    return (ASSETS / "xsec_ar_ion_phelps_lxcat.csv").read_text(encoding="utf-8")
