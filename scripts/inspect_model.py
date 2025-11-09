import sys
import os
import json
import torch


def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def inspect_model(path):
    print(f"Inspecting: {path}")
    exists = os.path.exists(path)
    print(f"Exists: {exists}")
    if not exists:
        return
    st = os.stat(path)
    print(f"Size: {fmt_bytes(st.st_size)} ({st.st_size} bytes)")
    print(f"Modified: {os.path.getmtime(path)}")

    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to torch.load: {e}")
        return

    # Detect common packaging: full checkpoint vs state_dict
    if isinstance(data, dict):
        # If wrapped (e.g., {'model_state_dict': ...})
        if "model_state_dict" in data:
            sd = data["model_state_dict"]
            print("Detected wrapper key 'model_state_dict'.")
        else:
            # assume this dict is the state_dict itself
            sd = data
    else:
        print(f"Loaded object type: {type(data)}")
        sd = None

    if sd is None:
        print("No state dict found to inspect.")
        return

    total_params = 0
    print("Parameters:")
    for k, v in sd.items():
        if hasattr(v, "shape"):
            num = int(torch.tensor(v).numel())
            total_params += num
            print(f"  {k}: {tuple(v.shape)} ({num})")
        else:
            print(f"  {k}: type={type(v)}")

    print(f"Total parameters: {total_params}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/best_model.pt"
    inspect_model(model_path)

    # print training_state.json if present
    ts_path = "training_state.json"
    print("\ntraining_state.json:")
    if os.path.exists(ts_path):
        try:
            with open(ts_path, "r", encoding="utf-8") as f:
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Failed to read {ts_path}: {e}")
    else:
        print("  not found")
