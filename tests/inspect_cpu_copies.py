import json
import sys
import gzip
from collections import defaultdict

def find_call_stacks(file_path):
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
    else:
        with open(file_path, "r") as f:
            data = json.load(f)

    events = data.get("traceEvents", [])
    
    stacks = defaultdict(float)
    counts = defaultdict(int)
    
    for event in events:
        name = event.get("name", "")
        if name in ["aten::copy_", "aten::to", "aten::_to_copy"]:
            dur = event.get("dur", 0) / 1000.0  # ms
            stack = event.get("args", {}).get("CpuStack", None)
            
            stack_key = "Unknown"
            if stack:
                # Find the first frame not from torch internals
                for frame in stack:
                    if 'torch' not in frame and 'site-packages' not in frame:
                        stack_key = frame
                        break
                if stack_key == "Unknown":
                    stack_key = stack[0]
                    
            stacks[stack_key] += dur
            counts[stack_key] += 1
            
    print(f"Aggregated CPU PyTorch Copy operations")
    for frame, dur in sorted(stacks.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"\n{dur:8.2f} ms ({counts[frame]:5d} calls) : {frame}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_call_stacks(sys.argv[1])
    else:
        print("Usage: python inspect_cpu_copies.py <trace.json>")
