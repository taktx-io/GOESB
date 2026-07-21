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
    for field in ("npu", "storage", "firmware", "cooling"):
        assert field in env
        assert field in env["unavailable"]

    if env["gpu"] is None:
        assert "gpu" in env["unavailable"]
    if env["power"] is None:
        assert "power" in env["unavailable"]


def test_capture_environment_is_json_serializable():
    import json

    json.dumps(capture_environment())
