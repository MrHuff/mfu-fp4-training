import json
import sys
import gzip

def find_call_stacks(file_path, target_names):
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
    else:
        with open(file_path, "r") as f:
            data = json.load(f)

    events = data.get("traceEvents", [])
    
    # We want to find the events with specific names and print their "args.CpuStack" if present
    for event in events:
        name = event.get("name", "")
        if name in target_names:
            dur = event.get("dur", 0) / 1000.0  # ms
            args = event.get("args", {})
            stack = args.get("CpuStack", None)
            
            # Torchtitan trace stacks might be under "args", check format:
            print(f"[{dur:.2f} ms] Found '{name}'")
            if stack:
                for idx, frame in enumerate(stack):
                    print(f"  {idx}: {frame}")
            
            # Additionally, print full args to see what context PyTorch gives us
            print(f"  Args: {json.dumps(args, indent=2)[:500]}")
            
    print("Completed scanning.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trace_file = sys.argv[1]
        find_call_stacks(trace_file, ["aten::_local_scalar_dense", "aten::item", "cudaStreamSynchronize"])
    else:
        print("Usage: python find_stacks.py <trace.json>")
