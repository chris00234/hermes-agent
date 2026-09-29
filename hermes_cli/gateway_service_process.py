"""Resolve gateway workers behind service-owned logging wrappers."""

from pathlib import Path


def _is_service_wrapper(argv: list[str]) -> bool:
    from gateway.status import inline_bootstrap_argv

    # Store launchers run the wrapper module through an inline bootstrap (``python -I -c <src> …``).
    argv = inline_bootstrap_argv(argv) or argv
    return bool(argv) and (
        Path(argv[0]).name == "osascript"
        or argv[1:3] == ["-m", "hermes_cli.stderr_timestamp"]
    )


def gateway_pid_for_service(pid: int) -> int | None:
    """Unwrap known service wrappers; refuse an ambiguous or unreadable worker."""
    import psutil
    from gateway.status import looks_like_gateway_command_line

    try:
        process = psutil.Process(pid)
        if not _is_service_wrapper(process.cmdline()):
            return pid
        gateways = []
        for child in process.children(recursive=True):
            try:
                argv = child.cmdline()
            except psutil.NoSuchProcess:
                continue
            if not _is_service_wrapper(argv) and looks_like_gateway_command_line(" ".join(argv)):
                gateways.append(child.pid)
        return gateways[0] if len(gateways) == 1 else None
    except psutil.NoSuchProcess:
        return pid
    except psutil.Error:
        return None
