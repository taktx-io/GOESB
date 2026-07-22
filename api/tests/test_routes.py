from conftest import make_signed_result


def test_list_profiles_returns_real_committed_profiles(client):
    r = client.get("/profiles")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["profiles"]}
    assert "whisper-medium-en-batch" in ids


def test_get_profile_returns_full_document(client):
    r = client.get("/profiles/whisper-medium-en-batch")
    assert r.status_code == 200
    assert r.json()["id"] == "whisper-medium-en-batch"
    assert "metrics" in r.json()


def test_get_unknown_profile_404s(client):
    r = client.get("/profiles/no-such-profile")
    assert r.status_code == 404


def test_list_packs_excludes_placeholder_pack(client):
    r = client.get("/packs")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["packs"]}
    assert "example-librispeech-en-batch" in ids
    # this pack's committed sha256 is a documented all-zero placeholder
    # (real audio not yet fetchable — consent-gated dataset) and must not
    # be served as if it were a verified asset.
    assert "example-common-voice-nl-batch" not in ids


def test_get_pack_returns_full_document(client):
    r = client.get("/packs/example-librispeech-en-batch")
    assert r.status_code == 200
    assert r.json()["id"] == "example-librispeech-en-batch"


def test_leaderboards_empty_before_any_ingest(client):
    r = client.get("/leaderboards")
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_leaderboards_reflects_ingested_results(client):
    result = make_signed_result(client)
    client.post("/benchmark", json=result)

    r = client.get("/leaderboards")
    assert r.status_code == 200
    ids = {entry["id"] for entry in r.json()["results"]}
    assert result["payload_sha256"] in ids


def test_hardware_empty_before_any_ingest(client):
    r = client.get("/hardware")
    assert r.status_code == 200
    assert r.json()["hardware"] == []


def test_hardware_reflects_ingested_results(client):
    result = make_signed_result(client)
    client.post("/benchmark", json=result)

    r = client.get("/hardware")
    assert r.status_code == 200
    assert len(r.json()["hardware"]) == 1
    assert r.json()["hardware"][0]["os_system"] == "Linux"
    assert r.json()["hardware"][0]["result_count"] == 1
