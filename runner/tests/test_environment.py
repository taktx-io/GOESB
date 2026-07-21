from oesb_runner.environment import capture_environment


def test_capture_environment_has_core_fields():
    env = capture_environment()
    assert env["schema_version"]
    assert "os" in env and "python" in env
