import sys
import os
from pathlib import Path

def setup_paths():
    # Add holohub path
    workspace_holohub_path = Path('/workspace/holohub')
    if workspace_holohub_path not in sys.path:
        sys.path += [
            str(workspace_holohub_path),
            str(workspace_holohub_path / 'operators'),
            str(workspace_holohub_path / 'applications'),
            str(workspace_holohub_path / 'tutorials'),
            str(workspace_holohub_path / 'pkg'),
        ]
        print(f"Added {workspace_holohub_path} to sys.path")


if __name__ == "__main__":
    setup_paths()
    print("Paths have been set up. sys.path now contains:")
    for path in sys.path:
        print(f"  {path}")