import sys
import os

# Add /workspace/holohub to the Python path
workspace_holohub_path = '/workspace/holohub'
if workspace_holohub_path not in sys.path:
    sys.path.append(workspace_holohub_path)

# Add ./operators to the Python path
current_dir_operators_path = os.path.dirname(__file__)
if current_dir_operators_path not in sys.path:
    sys.path.append(current_dir_operators_path)