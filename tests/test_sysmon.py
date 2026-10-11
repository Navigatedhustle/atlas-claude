from atlas import sysmon


def test_offline_window(monkeypatch):
    m = sysmon.SysMon()
    routes = {"v": ["eth0"]}
    tx = {"v": 1000}
    monkeypatch.setattr(sysmon, "_default_route", lambda: routes["v"])
    monkeypatch.setattr(sysmon, "_ifaces", lambda: {"eth0": {"tx": tx["v"], "rx": 0}, "lo": {"tx": 999999, "rx": 0}})
    m.reset()
    tx["v"] = 1500
    n = m.network()
    assert not n["offline"] and n["tx_bytes"] == 500 and "tx_since_offline" not in n
    routes["v"] = []                      # cable pulled
    n = m.network()
    assert n["offline"] and n["tx_since_offline"] == 0
    tx["v"] = 1500                        # nothing leaves the box
    assert m.network()["tx_since_offline"] == 0
    tx["v"] = 1600                        # if something did, it is counted, not hidden
    assert m.network()["tx_since_offline"] == 100
    routes["v"] = ["eth0"]
    assert "tx_since_offline" not in m.network()


def test_loopback_ignored(monkeypatch):
    monkeypatch.setattr(sysmon, "_ifaces", lambda: {"lo": {"tx": 5, "rx": 0}, "docker0": {"tx": 7, "rx": 0}})
    monkeypatch.setattr(sysmon, "_default_route", lambda: [])
    m = sysmon.SysMon()
    monkeypatch.setattr(sysmon, "_ifaces", lambda: {"lo": {"tx": 50000, "rx": 0}, "docker0": {"tx": 70000, "rx": 0}})
    assert m.network()["tx_bytes"] == 0
