import pytest

from oesb_runner import energy


def test_read_rapl_uj_missing_root_returns_none(tmp_path):
    assert energy.read_rapl_uj(root=tmp_path / "no-such-powercap") is None


def test_read_rapl_uj_sums_domains(tmp_path):
    root = tmp_path / "powercap"
    for domain, value in [("intel-rapl:0", "1000000"), ("intel-rapl:1", "500000")]:
        domain_dir = root / domain
        domain_dir.mkdir(parents=True)
        (domain_dir / "energy_uj").write_text(value)
    assert energy.read_rapl_uj(root=root) == pytest.approx(1_500_000.0)


def test_read_rapl_uj_ignores_unreadable_domain(tmp_path):
    root = tmp_path / "powercap"
    good = root / "intel-rapl:0"
    good.mkdir(parents=True)
    (good / "energy_uj").write_text("42")
    bad = root / "intel-rapl:1"
    bad.mkdir(parents=True)
    (bad / "energy_uj").write_text("not-a-number")
    assert energy.read_rapl_uj(root=root) == pytest.approx(42.0)


def test_read_rapl_uj_present_but_empty_returns_none(tmp_path):
    root = tmp_path / "powercap"
    root.mkdir()
    assert energy.read_rapl_uj(root=root) is None


def test_sample_hwmon_temp_c_missing_root_returns_none(tmp_path):
    assert energy.sample_hwmon_temp_c(root=tmp_path / "no-such-hwmon") is None


def test_sample_hwmon_temp_c_returns_peak_across_sensors(tmp_path):
    root = tmp_path / "hwmon"
    hwmon0 = root / "hwmon0"
    hwmon0.mkdir(parents=True)
    (hwmon0 / "temp1_input").write_text("45000")
    (hwmon0 / "temp2_input").write_text("62500")
    hwmon1 = root / "hwmon1"
    hwmon1.mkdir(parents=True)
    (hwmon1 / "temp1_input").write_text("38000")
    assert energy.sample_hwmon_temp_c(root=root) == pytest.approx(62.5)


def test_hwmon_available_reflects_sample_result(tmp_path):
    empty_root = tmp_path / "hwmon-empty"
    assert energy.hwmon_available(root=empty_root) is False

    populated_root = tmp_path / "hwmon-populated"
    hwmon0 = populated_root / "hwmon0"
    hwmon0.mkdir(parents=True)
    (hwmon0 / "temp1_input").write_text("30000")
    assert energy.hwmon_available(root=populated_root) is True
