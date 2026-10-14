"""Run untrusted, model-written programs (Python analysis code, LaTeX) in a throw-away sandbox.

Linux without root: ``unshare --user --map-root-user --mount --net --pid`` gives the program a private mount,
network and PID namespace. Inside it we mount fresh tmpfs over every directory a user could write or read private
data from (``/home``, ``/root``, ``/tmp``, ``/var/tmp``, ``/dev/shm``, ``/run``, ``/srv``, ``/mnt``, ``/media``), bind
the interpreter read-only at its original path and the work directory read-write at ``/home/sbx/work``, then drop
every capability (``setpriv``: empty bounding set, no_new_privs) before exec — otherwise the namespace root could
unmount those layers. The program gets a minimal environment (no tokens), no network (empty net namespace), and —
when ``systemd-run --user --scope`` works — a cgroup with memory, task and CPU limits; always a wall-clock timeout
and ``RLIMIT_FSIZE``.

Without ``unshare`` the sandbox is unavailable and :func:`run` refuses unless ``allow_unsandboxed=True`` (tests).
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HIDE = ("/home", "/root", "/tmp", "/var/tmp", "/dev/shm", "/run", "/srv", "/mnt", "/media")
WORK = "/home/sbx/work"
SAFE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/sbx", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "NUMEXPR_MAX_THREADS": "2",
            "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TEXMFVAR": "/home/sbx/.texmf-var"}


@dataclass
class Result:
    returncode: Optional[int]
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool
    sandboxed: bool
    files: Dict[str, bytes]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@lru_cache(maxsize=1)
def unshare_ok() -> bool:
    if not shutil.which("unshare") or not shutil.which("setpriv") or not sys.platform.startswith("linux"):
        return False
    try:
        r = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork",
                            "sh", "-c", "mount -t tmpfs tmpfs /tmp"], capture_output=True, timeout=15)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def scope_ok() -> bool:
    if not shutil.which("systemd-run"):
        return False
    try:
        r = subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", "TasksMax=16", "true"],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def available() -> bool:
    return unshare_ok()


def _hidden(path: Path) -> bool:
    p = str(path)
    return any(p == h or p.startswith(h + "/") for h in HIDE)


@lru_cache(maxsize=8)
def python_roots(python: str) -> tuple:
    """Directories an interpreter needs: its venv, the base installation and the ``home`` of ``pyvenv.cfg``, both
    as written and resolved (tools such as uv link a ``cpython-3.11`` directory to ``cpython-3.11.15``). When all of
    them share one deep enough ancestor (e.g. a runtime's version directory) that ancestor is bound instead."""
    out = subprocess.run([python, "-c", "import sys; print(sys.prefix); print(sys.base_prefix)"],
                         capture_output=True, text=True, timeout=30, check=True).stdout.split()
    paths = {Path(python).parent.parent, Path(python).resolve().parent.parent, *map(Path, out)}
    cfg = Path(python).parent.parent / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, val = line.partition("=")
            if key.strip() == "home" and val.strip():
                paths.add(Path(val.strip()).parent)
    paths |= {p.resolve() for p in list(paths)}
    common = Path(os.path.commonpath([str(p) for p in paths]))
    if len(common.parts) >= 5 and common != Path.home():
        return (str(common),)
    return tuple(sorted(str(p) for p in paths))


def _mount_script(binds: List[str], hold: str, work_hold: str, env: Dict[str, str]) -> str:
    """Shell script run as root of the new user namespace: hide private dirs, re-expose only what is needed."""
    temp_root = next((h for h in HIDE if hold == h or hold.startswith(h + "/")), None)
    lines = ["set -e"]
    for i, src in enumerate(binds):
        lines += [f'mkdir -p "{hold}/b{i}"', f'mount --bind "{src}" "{hold}/b{i}"',
                  f'mount -o remount,bind,ro "{hold}/b{i}"']
    for d in HIDE:
        if d != temp_root:
            lines.append(f'if [ -d "{d}" ]; then mount -t tmpfs -o size=256m,mode=755 tmpfs "{d}"; fi')
    for i, src in enumerate(binds):
        lines += [f'mkdir -p "{src}"', f'mount --bind "{hold}/b{i}" "{src}"']
    lines += [f'mkdir -p "{WORK}" /home/sbx/.texmf-var', f'mount --bind "{work_hold}" "{WORK}"']
    if temp_root:   # last: the hold and work dirs live under it
        lines.append(f'mount -t tmpfs -o size=256m,mode=1777 tmpfs "{temp_root}"')
    exports = " ".join(f"{k}='{v}'" for k, v in env.items())
    # The script runs as root of the user namespace with every capability (needed for the mounts). The program must
    # not keep them: with CAP_SYS_ADMIN it could unmount the tmpfs layers and reach the real /home, /run (Docker
    # socket) or /proc. setpriv empties the bounding set (so exec grants no capabilities) and sets no_new_privs.
    lines += [f'cd "{WORK}"',
              f'exec setpriv --no-new-privs --bounding-set=-all --inh-caps=-all --ambient-caps=-all env -i {exports} "$@"']
    return "\n".join(lines)


def _limits(fsize_mb: int, cpu_s: int, as_mb: Optional[int]):
    def apply():  # runs in the child before exec
        os.setsid()
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        if as_mb:
            resource.setrlimit(resource.RLIMIT_AS, (as_mb << 20, as_mb << 20))
    return apply


def run(argv: Sequence[str], files: Dict[str, bytes], timeout: float = 60, mem_mb: int = 3072, tasks: int = 128,
        cpu_quota: str = "200%", ro_binds: Sequence[str] = (), collect: Sequence[str] = (),
        allow_unsandboxed: bool = False, fsize_mb: int = 64) -> Result:
    """Run ``argv`` with ``files`` written into the work directory; return output and the ``collect``-ed files."""
    sandboxed = available()
    if not sandboxed and not allow_unsandboxed:
        raise RuntimeError("sandbox unavailable (needs Linux user namespaces: unshare --user --map-root-user)")
    base = Path(tempfile.mkdtemp(prefix="pa-sbx-"))
    work = base / "work"
    work.mkdir()
    try:
        for name, data in files.items():
            target = (work / name).resolve()
            if work.resolve() not in target.parents:
                raise ValueError(f"file name escapes the work dir: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if sandboxed:
            binds = [b for b in dict.fromkeys(str(Path(b)) for b in ro_binds) if _hidden(Path(b))]
            script = _mount_script(binds, str(base / "hold"), str(work), SAFE_ENV)
            cmd = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child",
                   "--mount-proc", "sh", "-c", script, "sh", *argv]
            as_mb: Optional[int] = mem_mb * 2
            if scope_ok():
                cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem_mb}M", "-p", "MemorySwapMax=0",
                       "-p", f"TasksMax={tasks}", "-p", f"CPUQuota={cpu_quota}", *cmd]
                as_mb = None
            cwd, env = None, dict(os.environ)      # the program itself gets only SAFE_ENV (env -i in the script)
        else:
            cmd, as_mb, cwd = list(argv), mem_mb * 2, str(work)
            env = {**SAFE_ENV, "HOME": str(work)}
        t0 = time.time()
        timed_out = False
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                preexec_fn=_limits(fsize_mb, int(timeout) + 5, as_mb))
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            out, err = proc.communicate()
        got = {}
        for name in collect:
            p = work / name
            if p.is_file():
                got[name] = p.read_bytes()
        return Result(None if timed_out else proc.returncode, out.decode("utf-8", "replace")[-200_000:],
                      err.decode("utf-8", "replace")[-50_000:], round(time.time() - t0, 3), timed_out, sandboxed, got)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def run_python(code: str, files: Dict[str, bytes], python: str, timeout: float = 60, **kw) -> Result:
    """Run ``code`` as ``main.py``. ``python`` may be a venv interpreter: its directory is resolved (symlinked
    parents are not visible in the sandbox) but the executable name is kept so the venv is still detected."""
    exe = str(Path(python).parent.resolve() / Path(python).name)
    roots = python_roots(exe) if available() else ()
    return run([exe, "main.py"], {**files, "main.py": code.encode("utf-8")}, timeout=timeout,
               ro_binds=roots, **kw)


def default_python() -> str:
    """Interpreter for model-written analysis code: ``BENCH_PYTHON`` or the current one."""
    return os.getenv("BENCH_PYTHON") or sys.executable
