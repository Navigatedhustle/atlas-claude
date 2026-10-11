"""Sandbox bootstrap: lock this process down, then exec the real command.

Usage: python sandbox_boot.py <workdir> <mem_mb> <cpu_s> -- <cmd...>

Unprivileged user namespaces are blocked on this machine (AppArmor), so `unshare -rn` and bwrap cannot
be used. Instead three kernel mechanisms are applied, all inherited across exec and impossible to undo:
  1. rlimits: address space (memory), CPU seconds, file size, open files.
  2. Landlock (LSM, ABI >= 1): the process may write only inside <workdir>; everything else, including the
     read-only database copy, is read-only. With ABI >= 4 all TCP bind/connect is also denied.
  3. seccomp-BPF: socket() is refused for every address family except AF_UNIX (so no IPv4/IPv6/packet/
     netlink sockets at all), io_uring is refused, and chmod/chown calls are refused.
"""
import ctypes
import os
import resource
import struct
import sys

libc = ctypes.CDLL(None, use_errno=True)
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2

# --- Landlock -------------------------------------------------------------------------------------------
SYS_LL_CREATE, SYS_LL_ADD, SYS_LL_RESTRICT = 444, 445, 446
FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1, 2, 4, 8
FS_REMOVE_DIR, FS_REMOVE_FILE, FS_MAKE_CHAR, FS_MAKE_DIR = 16, 32, 64, 128
FS_MAKE_REG, FS_MAKE_SOCK, FS_MAKE_FIFO, FS_MAKE_BLOCK, FS_MAKE_SYM = 256, 512, 1024, 2048, 4096
FS_REFER, FS_TRUNCATE = 8192, 16384
NET_BIND_TCP, NET_CONNECT_TCP = 1, 2


def landlock(workdir: str) -> int:
    abi = libc.syscall(SYS_LL_CREATE, None, 0, 1)
    if abi < 1:
        raise OSError("landlock unavailable")
    write = (FS_WRITE_FILE | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_CHAR | FS_MAKE_DIR | FS_MAKE_REG |
             FS_MAKE_SOCK | FS_MAKE_FIFO | FS_MAKE_BLOCK | FS_MAKE_SYM)
    if abi >= 2:
        write |= FS_REFER
    if abi >= 3:
        write |= FS_TRUNCATE
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0
    attr = struct.pack("QQ", write, net) if abi >= 4 else struct.pack("Q", write)
    fd = libc.syscall(SYS_LL_CREATE, ctypes.c_char_p(attr), len(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")

    def allow(path, access):
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = struct.pack("=Qi", access, pfd)  # struct landlock_path_beneath_attr is packed
        if libc.syscall(SYS_LL_ADD, fd, 1, ctypes.c_char_p(rule), 0) != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
        os.close(pfd)

    allow(workdir, write)
    allow("/dev/null", FS_WRITE_FILE | (FS_TRUNCATE if abi >= 3 else 0))
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "no_new_privs")
    if libc.syscall(SYS_LL_RESTRICT, fd, 0) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    return abi


# --- seccomp ---------------------------------------------------------------------------------------------
AUDIT_ARCH = {"aarch64": 0xC00000B7, "x86_64": 0xC000003E}
NR = {"aarch64": {"socket": 198, "socketpair": 199, "io_uring_setup": 425, "fchmod": 52, "fchmodat": 53,
                  "fchmodat2": 452, "fchown": 55, "fchownat": 54},
      "x86_64": {"socket": 41, "socketpair": 53, "io_uring_setup": 425, "chmod": 90, "fchmod": 91, "fchmodat": 268,
                 "fchmodat2": 452, "chown": 92, "fchown": 93, "lchown": 94, "fchownat": 260}}
BPF_LD_W_ABS, BPF_JEQ_K, BPF_RET_K = 0x20, 0x15, 0x06
RET_ALLOW, RET_KILL, RET_ERRNO = 0x7FFF0000, 0x80000000, 0x00050000
EACCES, AF_UNIX = 13, 1


def seccomp():
    arch = os.uname().machine
    nr = NR[arch]
    deny = [v for k, v in nr.items() if k not in ("socket", "socketpair")]
    prog = []
    ins = lambda code, jt, jf, k: prog.append(struct.pack("HBBI", code, jt, jf, k))
    # [0] arch check
    ins(BPF_LD_W_ABS, 0, 0, 4)
    ins(BPF_JEQ_K, 1, 0, AUDIT_ARCH[arch])
    ins(BPF_RET_K, 0, 0, RET_KILL)
    ins(BPF_LD_W_ABS, 0, 0, 0)  # syscall number
    for n in deny:
        ins(BPF_JEQ_K, 0, 1, n)
        ins(BPF_RET_K, 0, 0, RET_ERRNO | EACCES)
    for n in (nr["socket"], nr["socketpair"]):
        ins(BPF_JEQ_K, 0, 4, n)
        ins(BPF_LD_W_ABS, 0, 0, 16)  # args[0] low word = address family
        ins(BPF_JEQ_K, 1, 0, AF_UNIX)
        ins(BPF_RET_K, 0, 0, RET_ERRNO | EACCES)
        ins(BPF_LD_W_ABS, 0, 0, 0)  # reload nr (also the fall-through target)
    ins(BPF_RET_K, 0, 0, RET_ALLOW)
    buf = ctypes.create_string_buffer(b"".join(prog))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fp = SockFprog(len(prog), ctypes.cast(buf, ctypes.c_void_p))
    libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fp), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "seccomp")


def main():
    workdir, mem_mb, cpu_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cmd = sys.argv[sys.argv.index("--") + 1:]
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**20, 64 * 2**20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    landlock(os.path.abspath(workdir))
    seccomp()
    os.chdir(workdir)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
