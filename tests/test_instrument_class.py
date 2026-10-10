"""The run's instrument class for the class-gated stages (PassConfig.instrument_class)."""
from peaky.assignment import assign as A
from peaky.assignment import passes
from peaky.chem.resolution import Resolution


def test_snapshot_type_names_the_class():
    assert A.instrument_class_of("orbi", None) == "orbitrap"
    assert A.instrument_class_of("Orbitrap", None) == "orbitrap"
    assert A.instrument_class_of("tof", Resolution(coef=2e-7, exponent=1.5, source="measured")) == "tof"


def test_a_measured_width_model_decides_without_a_type():
    orbi = Resolution(coef=1.2e-7, exponent=1.65, n_peaks=40, source="measured")
    tof = Resolution(coef=1.0 / 10_000, exponent=1.0, n_peaks=40, source="measured")
    assert A.instrument_class_of(None, orbi) == "orbitrap"
    assert A.instrument_class_of(None, tof) == "tof"


def test_a_declared_number_is_not_evidence_of_the_class():
    assert A.instrument_class_of(None, Resolution.from_r(250_000)) is None
    assert A.instrument_class_of(None, None) is None


def test_the_fields_are_runtime_and_default_off():
    cfg = passes.PassConfig()
    assert cfg.instrument_class is None and cfg.trace_sample is False
    assert {"instrument_class", "trace_sample"} <= set(passes.PassConfig.RUNTIME_FIELDS)
