"""A minimal 'sandboxed' Python execution tool: separate process, isolated mode (-I), empty environment, fresh temp working dir,
rlimits (CPU 2 s, address space 512 MB, max file size 1 MB), 5 s wall-clock timeout, output truncated to 2000 chars.
NOT a security boundary: no filesystem, network or syscall isolation (that needs containers / seccomp / gVisor-style VMs)."""
import os, resource, subprocess, sys, tempfile
def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2)); resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20,) * 2)
def run_python(code, timeout=5):
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, env={}, capture_output=True, text=True, timeout=timeout, preexec_fn=_limits)
            out = (p.stdout + (("\n[stderr] " + p.stderr.strip().splitlines()[-1]) if p.stderr.strip() else ""))[:2000]
            return {"returncode": p.returncode, "output": out.strip()}
        except subprocess.TimeoutExpired: return {"returncode": None, "output": f"[killed: wall-clock timeout {timeout} s]"}
