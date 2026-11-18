"""Expose this kernel's prebuilt anywidget frontend to the user's JupyterLab server."""
import json
from pathlib import Path

import anywidget
from jupyter_core.paths import jupyter_data_dir


def enable():
    source = Path(anywidget.__file__).resolve().parent / "labextension"
    package = json.loads((source / "package.json").read_text())
    bundle = source / package["jupyterlab"]["_build"]["load"]
    if not bundle.is_file():
        raise SystemExit(f"Missing anywidget browser bundle at {bundle}")
    target = Path(jupyter_data_dir()) / "labextensions/anywidget"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and not target.exists():
        target.unlink()
    try:
        target.symlink_to(source, target_is_directory=True)
    except FileExistsError:
        if target.resolve() != source:
            existing = target / "package.json"
            if (not existing.is_file() or json.loads(existing.read_text()).get("version") != package["version"]
                    or not (target / bundle.relative_to(source)).is_file()):
                raise SystemExit(f"An existing installation at {target} differs from {source}. "
                                 "Check this installation in the OnDemand JupyterLab server environment.")
    print(f"anywidget {package['version']} is available at {target}")
    print("Stop and relaunch the OnDemand JupyterLab session, then refresh the browser tab.")


if __name__ == "__main__":
    enable()
