"""Run a Python script/module with a process-local Python socket audit guard.

Usage: python tests/run_without_network.py --self-test
       python tests/run_without_network.py -m pip install --no-index ...
       python tests/run_without_network.py run.py --data data --out out-check/qa

This is QA instrumentation, not a firewall or a security sandbox. It blocks
the listed Python socket audit events, including loopback, in this process.
It does not block child processes or native network calls outside those hooks.
Interpreter startup before this runner, such as sitecustomize, is not covered.
"""

import os
import runpy
import socket
import sys


BLOCKED_EVENTS = frozenset({
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.getnameinfo",
    "socket.sendto",
    "socket.sendmsg",
})


class NetworkBlockedError(RuntimeError):
    """A socket operation was rejected before its audited system call."""


class SocketAuditGuard:
    def __init__(self):
        self.blocked = 0

    def __call__(self, event, _arguments):
        if event in BLOCKED_EVENTS:
            self.blocked += 1
            raise NetworkBlockedError("Python socket operation blocked by QA guard")

    def report(self):
        if self.blocked:
            print(f"Blocked Python socket attempts: {self.blocked}", file=sys.stderr)


def self_test(guard):
    """Exercise a real connect API; the audit exception prevents the syscall."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        try:
            probe.connect(("127.0.0.1", 9))
        except NetworkBlockedError:
            if guard.blocked == 1:
                print("PASS: loopback connect blocked before the socket call")
                return 0
        except OSError:
            pass
    print("FAIL: socket audit guard did not intercept the probe", file=sys.stderr)
    return 1


def main():
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if arguments else 2
    if arguments[0] == "--self-test" and len(arguments) != 1:
        print("--self-test takes no arguments", file=sys.stderr)
        return 2
    is_module = arguments[0] == "-m"
    if is_module and (len(arguments) < 2 or arguments[1].startswith("-")):
        print("-m requires a module name", file=sys.stderr)
        return 2
    if arguments[0].startswith("-") and arguments[0] not in ("-m", "--self-test"):
        print("Expected --self-test, -m MODULE, or SCRIPT", file=sys.stderr)
        return 2

    guard = SocketAuditGuard()
    sys.addaudithook(guard)
    if arguments == ["--self-test"]:
        try:
            return self_test(guard)
        finally:
            guard.report()

    target_arguments = arguments[1:] if is_module else arguments
    target = target_arguments[0]
    sys.argv[:] = target_arguments
    # Replace the runner's tests/ directory with the entry a normal invocation
    # supplies. Respect -P / -I: they deliberately omit that unsafe path entry.
    if not sys.flags.safe_path:
        entry = os.getcwd() if is_module else os.path.dirname(os.path.realpath(target))
        sys.path[:1] = [entry]

    exit_code = 0
    try:
        if is_module:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(target, run_name="__main__")
    except NetworkBlockedError:
        exit_code = 1
    except SystemExit as error:
        exit_code = error.code
    finally:
        guard.report()
    # A target may catch the guard exception; it must not turn the QA run green.
    if guard.blocked and (exit_code is None or isinstance(exit_code, int) and exit_code == 0):
        return 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
