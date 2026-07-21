from oesb_runner import environment, hashing


def test_canonical_asset_sha256_ignores_declared_sha256_field():
    a = {"id": "x", "version": "1.0.0", "sha256": "aaaa"}
    b = {"id": "x", "version": "1.0.0", "sha256": "bbbb"}
    assert hashing.canonical_asset_sha256(a) == hashing.canonical_asset_sha256(b)


def test_canonical_asset_sha256_is_order_independent():
    a = {"id": "x", "version": "1.0.0"}
    b = {"version": "1.0.0", "id": "x"}
    assert hashing.canonical_asset_sha256(a, exclude=()) == hashing.canonical_asset_sha256(b, exclude=())


def test_canonical_asset_sha256_changes_with_content():
    a = {"id": "x", "version": "1.0.0"}
    b = {"id": "x", "version": "1.0.1"}
    assert hashing.canonical_asset_sha256(a, exclude=()) != hashing.canonical_asset_sha256(b, exclude=())


def test_sha256_dir_is_deterministic(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    first = hashing.sha256_dir(tmp_path)
    second = hashing.sha256_dir(tmp_path)
    assert first == second


def test_sha256_dir_changes_when_content_changes(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    before = hashing.sha256_dir(tmp_path)
    (tmp_path / "a.txt").write_text("goodbye")
    after = hashing.sha256_dir(tmp_path)
    assert before != after


def test_sha256_module_source_matches_file_hash():
    expected = hashing.sha256_file(environment.__file__)
    assert hashing.sha256_module_source(environment) == expected
