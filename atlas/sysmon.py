"""Real system telemetry: network egress, offline detection, memory, GPU. Nothing here is simulated."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import psutil

VIRTUAL_PREFIXES = ("lo", "docker", "veth", "br-", "virbr", "tailscale", "zt", "tun", "tap")


def _ifaces() -> dict[str, dict]:
    out = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, rest = line.split(":", 1)
        name = name.strip()
        f = rest.split()
        out[name] = {"rx": int(f[0]), "tx": int(f[8])}
    return out


def _physical(name: str) -> bool:
    return not name.startswith(VIRTUAL_PREFIXES)


def _carrier(name: str) -> bool:
    try:
        return Path(f"/sys/class/net/{name}/carrier").read_text().strip() == "1"
    except OSError:
        return False


def _default_route() -> list[str]:
    devs = []
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            f = line.split()
            if f[1] == "00000000" and (int(f[3], 16) & 0x1):  # destination 0.0.0.0, RTF_UP
                devs.append(f[0])
    except OSError:
        pass
    try:  # IPv6 default routes
        for line in Path("/proc/net/ipv6_route").read_text().splitlines():
            f = line.split()
            if f[0] == "0" * 32 and f[1] == "00" and f[9] != "lo" and int(f[8], 16) & 0x1 and not f[9].startswith(VIRTUAL_PREFIXES):
                devs.append(f[9])
    except OSError:
        pass
    return sorted(set(devs))


class SysMon:
    def __init__(self):
        self.reset()
        self._gpu = {"util": None, "t": 0.0}

    def reset(self):
        snap = _ifaces()
        self.base = {k: v for k, v in snap.items() if _physical(k)}
        self.t0 = time.time()
        self.was_offline = None
        self.off_base = None
        self.off_t = None

    def network(self) -> dict:
        snap = _ifaces()
        tx = rx = 0
        for k, v in snap.items():
            if not _physical(k):
                continue
            b = self.base.get(k, {"tx": v["tx"], "rx": v["rx"]})
            tx += max(v["tx"] - b["tx"], 0)
            rx += max(v["rx"] - b["rx"], 0)
        routes = _default_route()
        links = [k for k in snap if _physical(k) and _carrier(k)]
        offline = not routes
        # When the box goes offline, start a second, clearly labelled window: bytes sent since going offline.
        if offline and not self.was_offline:
            self.off_base, self.off_t = tx, time.time()
        if not offline:
            self.off_base, self.off_t = None, None
        self.was_offline = offline
        out = {"offline": offline, "default_route_ifaces": routes, "links_up": links,
               "tx_bytes": tx, "rx_bytes": rx, "since": self.t0}
        if offline and self.off_base is not None:
            out["tx_since_offline"] = tx - self.off_base
            out["offline_since"] = self.off_t
        return out

    def gpu(self) -> dict:
        now = time.time()
        if now - self._gpu["t"] > 2.0:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                u, temp, pw = [x.strip() for x in r.stdout.strip().split(",")]
                self._gpu = {"util": float(u), "temp": float(temp), "power": None if "N/A" in pw else float(pw), "t": now}
            except Exception:
                self._gpu["t"] = now
        return {k: v for k, v in self._gpu.items() if k != "t"}

    def _gpu_apps(self) -> dict[int, int]:
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2)
            out = {}
            for line in r.stdout.strip().splitlines():
                pid, mem = [x.strip() for x in line.split(",")]
                if mem.isdigit():
                    out[int(pid)] = int(mem) * 2**20
            return out
        except Exception:
            return {}

    def memory(self, pidfiles_dir: Path) -> dict:
        """System memory plus Atlas's own footprint (CPU RSS + CUDA allocations of the processes we started)."""
        vm = psutil.virtual_memory()
        now = time.time()
        if now - getattr(self, "_apps_t", 0) > 3:
            self._apps, self._apps_t = self._gpu_apps(), now
        own = 0
        pids = [os.getpid()]
        for pf in pidfiles_dir.glob("*.pid"):
            try:
                p = psutil.Process(int(pf.read_text().strip()))
                pids.append(p.pid)
                pids.extend(c.pid for c in p.children(recursive=True))
            except Exception:
                pass
        for pid in set(pids):
            # Anonymous resident memory (excludes mmapped weight files that the page cache can drop)
            # plus CUDA allocations reported by the driver for that process.
            try:
                anon = 0
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("RssAnon:"):
                        anon = int(line.split()[1]) * 1024
                own += anon + self._apps.get(pid, 0)
            except Exception:
                pass
        return {"total_gb": vm.total / 2**30, "used_gb": (vm.total - vm.available) / 2**30,
                "available_gb": vm.available / 2**30, "atlas_gb": own / 2**30}
