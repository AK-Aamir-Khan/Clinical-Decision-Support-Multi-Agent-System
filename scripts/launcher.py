"""One-click launcher for MedAgent-CDSS (used by start_medagent.bat / stop_medagent.bat).

What it does, in order:
  1. Reads the LLM settings from the project's own config (.env via app.config),
     so the Ollama URL and model name are never hard-coded here.
  2. Detects a running Ollama server; starts `ollama serve` only if none is running.
  3. Makes sure the configured model (gemma3:4b) is available; pulls it if missing.
  4. Starts the existing Streamlit app exactly as documented:
         python -m streamlit run app/main.py
  5. Waits until the app answers its health check, then opens the browser.
  6. Records ONLY the processes it started, so `--stop` never touches anything else.

Usage (from the project root, with the virtual environment active):
    python scripts/launcher.py              # start (default port 8501, Streamlit's default)
    python scripts/launcher.py --stop       # stop what the launcher started
    python scripts/launcher.py --no-browser # start without opening a browser
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE_FILE = ROOT / ".medagent_launcher.json"
LOG_DIR = ROOT / "logs"
APP_ENTRY = "app/main.py"
DEFAULT_PORT = 8501  # Streamlit's default; the project has no .streamlit/config.toml overriding it
IS_WINDOWS = os.name == "nt"


def say(msg: str) -> None:
    print(f"[MedAgent] {msg}", flush=True)


def fail(msg: str, code: int = 1) -> int:
    print(f"\n[MedAgent] ERROR: {msg}\n", flush=True)
    return code


# ----------------------------------------------------------------------------- HTTP helpers
def http_get(url: str, timeout: float = 3.0):
    """Return (status, body) or (None, None) if nothing is listening."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def ollama_root(base_url: str) -> str:
    """http://localhost:11434/v1 -> http://localhost:11434 (Ollama's native API root)."""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def ollama_running(root: str) -> bool:
    status, _ = http_get(f"{root}/api/version")
    return status == 200


def ollama_models(root: str) -> list:
    status, body = http_get(f"{root}/api/tags", timeout=10)
    if status != 200 or not body:
        return []
    try:
        return [m.get("name", "") for m in json.loads(body).get("models", [])]
    except json.JSONDecodeError:
        return []


def model_available(root: str, model: str) -> bool:
    names = ollama_models(root)
    wanted = {model, model if ":" in model else f"{model}:latest"}
    return any(n in wanted for n in names)


def find_ollama_exe():
    exe = shutil.which("ollama")
    if exe:
        return exe
    if IS_WINDOWS:  # default per-user install location of the Ollama Windows app
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            return str(candidate)
    return None


# ----------------------------------------------------------------------------- process helpers
def background_flags() -> dict:
    """Start a process without a window, detached from this console (Windows)."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def process_alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def process_name(pid: int) -> str:
    """Image name / command line of a PID, used to avoid killing a recycled PID."""
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout.strip()
        return out.split(",")[0].strip('"').lower() if out.startswith('"') else ""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().lower()
    except OSError:
        return ""


def kill_tree(pid: int) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            if os.getpgid(pid) != os.getpgid(0):  # own process group: safe to stop the whole group
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:  # shares our group (e.g. the terminal's): stop only that process
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def streamlit_healthy(port: int) -> bool:
    status, body = http_get(f"http://localhost:{port}/_stcore/health")
    return status == 200 and "ok" in (body or "")


# ----------------------------------------------------------------------------- state file
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def clear_state() -> None:
    try:
        STATE_FILE.unlink()
    except OSError:  # e.g. file locked - blank it instead so --stop reports "nothing to stop"
        try:
            STATE_FILE.write_text("{}", encoding="utf-8")
        except OSError:
            pass


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- steps
def ensure_ollama(settings) -> tuple:
    """Return (ok, started_pid_or_None)."""
    root = ollama_root(settings.openai_base_url)
    model = settings.openai_model
    say(f"LLM configured: {model} via {settings.openai_base_url}")

    started_pid = None
    if ollama_running(root):
        say(f"Ollama is already running at {root} - not starting another server.")
    else:
        exe = find_ollama_exe()
        if not exe:
            fail("Ollama is not running and the 'ollama' command was not found.\n"
                 "  Install Ollama from https://ollama.com (or start the Ollama app), then run this launcher again.")
            return False, None
        LOG_DIR.mkdir(exist_ok=True)
        log = open(LOG_DIR / "ollama.log", "a", encoding="utf-8")
        say(f"Starting Ollama server ({exe} serve) - log: logs\\ollama.log")
        proc = subprocess.Popen([exe, "serve"], stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **background_flags())
        started_pid = proc.pid
        for _ in range(60):
            if ollama_running(root):
                break
            if proc.poll() is not None:
                fail(f"Ollama exited immediately (exit code {proc.returncode}). See logs\\ollama.log.")
                return False, None
            time.sleep(1)
        else:
            fail("Ollama did not respond within 60 seconds. See logs\\ollama.log.")
            return False, started_pid
        say("Ollama is running.")

    if model_available(root, model):
        say(f"Model '{model}' is available.")
    else:
        exe = find_ollama_exe()
        if not exe:
            fail(f"Model '{model}' is not available and the 'ollama' command was not found to pull it.\n"
                 f"  Run:  ollama pull {model}")
            return False, started_pid
        say(f"Model '{model}' not found - downloading it now (one-time, about 3.3 GB)...")
        if subprocess.run([exe, "pull", model]).returncode != 0 or not model_available(root, model):
            fail(f"Could not download '{model}'. Check your internet connection and run:  ollama pull {model}")
            return False, started_pid
        say(f"Model '{model}' downloaded.")
    return True, started_pid


def start(port: int, open_browser: bool) -> int:
    from app.config import get_settings  # the project's own configuration (.env)

    settings = get_settings()
    url = f"http://localhost:{port}"

    # Already running? Don't start a second copy - just open it.
    if streamlit_healthy(port):
        say(f"MedAgent-CDSS is already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return 0
    if port_in_use(port):
        return fail(f"Port {port} is already used by another program.\n"
                    f"  Close it, or start on another port:  start_medagent.bat --port 8502")

    state = {"port": port, "ollama_pid": None, "streamlit_pid": None}
    if settings.llm_enabled and settings.is_local:
        ok, ollama_pid = ensure_ollama(settings)
        state["ollama_pid"] = ollama_pid
        save_state(state)
        if not ok:
            return 1
    elif settings.llm_enabled:
        say(f"LLM configured as {settings.openai_model} (remote API) - no local Ollama check needed.")
    else:
        say("No LLM configured in .env (OPENAI_BASE_URL) - the app will run in rule-based mode.")

    say(f"Starting MedAgent-CDSS: streamlit run {APP_ENTRY} (port {port})")
    cmd = [sys.executable, "-m", "streamlit", "run", APP_ENTRY,
           "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    app = subprocess.Popen(cmd, cwd=str(ROOT))
    state["streamlit_pid"] = app.pid
    save_state(state)

    for _ in range(90):
        if streamlit_healthy(port):
            break
        if app.poll() is not None:
            stop_started(state)
            return fail(f"The app exited during startup (exit code {app.returncode}). See the messages above.")
        time.sleep(1)
    else:
        stop_started(state)
        return fail("The app did not become ready within 90 seconds. See the messages above.")

    say(f"MedAgent-CDSS is ready at {url}")
    if open_browser:
        webbrowser.open(url)
    say("Keep this window open while using the app. Press Ctrl+C here (or run stop_medagent.bat) to stop.")

    try:
        code = app.wait()
    except KeyboardInterrupt:
        say("Stopping...")
        code = 0
    stop_started(state)
    say("Stopped.")
    return code


def stop_started(state: dict) -> None:
    """Stop only the processes this launcher started (verified by PID and process name)."""
    expected = {"streamlit_pid": ("python", "streamlit"), "ollama_pid": ("ollama",)}
    for key, names in expected.items():
        pid = state.get(key)
        if not pid or not process_alive(pid):
            continue
        name = process_name(pid)
        if not any(n in name for n in names):
            say(f"Skipping PID {pid}: it is no longer a launcher process ({name or 'unknown'}).")
            continue
        kill_tree(pid)
        say(f"Stopped {key.replace('_pid', '')} (PID {pid}).")
    clear_state()


def stop() -> int:
    state = load_state()
    if not state:
        say("Nothing to stop: no processes were started by the launcher (or they already stopped).")
        return 0
    if not state.get("ollama_pid"):
        say("Ollama was already running before the launcher started, so it is left running.")
    stop_started(state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Start or stop MedAgent-CDSS (Streamlit + Ollama).")
    parser.add_argument("--stop", action="store_true", help="stop the processes started by the launcher")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="web port (default 8501)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()
    os.chdir(ROOT)
    return stop() if args.stop else start(args.port, not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
