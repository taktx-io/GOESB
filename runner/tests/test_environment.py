from oesb_runner.environment import capture_environment


def test_capture_environment_has_core_fields():
    env = capture_environment()
    assert env["schema_version"]
    assert "os" in env and "python" in env
    assert env["cpu"]["physical_cores"] is None or env["cpu"]["physical_cores"] >= 1
    assert env["ram"]["total_mb"] > 0


def test_capture_environment_never_silently_omits_unprobed_fields():
    env = capture_environment()
    # Fields with no probe on this platform must be null, not missing, and
    # explained in `unavailable` (docs/specs/environment-capture.md).
    for field in ("npu", "storage", "firmware"):
        assert field in env
        assert field in env["unavailable"]

    # gpu/power/cooling have real probes and may or may not fire depending on
    # the machine (NVIDIA GPU present, on battery, hwmon sensors present) —
    # null must always come with a reason, but non-null is equally valid.
    for field in ("gpu", "power", "cooling"):
        assert field in env
        if env[field] is None:
            assert field in env["unavailable"]


def test_capture_environment_is_json_serializable():
    import json

    json.dumps(capture_environment())
