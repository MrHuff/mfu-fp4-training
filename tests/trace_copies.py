import sys
import json
import gzip
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python trace_copies.py <trace_file.json.gz>")
        sys.exit(1)
        
    trace_file = sys.argv[1]
    
    with gzip.open(trace_file, 'rt') as f:
        data = json.load(f)
        
    events = data.get('traceEvents', [])
    
    # 1. Find all correlation IDs for direct_copy GPU kernels
    gpu_copies = []
    gpu_correlations = set()
    for e in events:
        name = e.get('name', '')
        if 'direct_copy' in name and e.get('ph') == 'X':
            args = e.get('args', {})
            corr = args.get('correlation')
            if corr is not None:
                gpu_copies.append({
                    'dur': e.get('dur', 0),
                    'name': name,
                    'correlation': corr
                })
                gpu_correlations.add(corr)
                
    # 2. Find CPU cudaLaunchKernel / cudaMemcpy events that match these correlations
    launch_events = {}
    for e in events:
        if e.get('cat') == 'cuda_runtime' or 'Launch' in e.get('name', ''):
            corr = e.get('args', {}).get('correlation')
            if corr in gpu_correlations:
                launch_events[corr] = e
                
    # 3. Build time-intervals for Operators to see what encapsulates the launch
    # Collect all Operators/cpu_ops
    cpu_ops = defaultdict(list)
    for e in events:
        if e.get('ph') == 'X' and e.get('cat', '') in ('cpu_op', 'user_annotation', 'python_function', 'Operator'):
            tid = e.get('tid')
            cpu_ops[tid].append(e)
            
    # Sort them by timestamp
    for tid in cpu_ops:
        cpu_ops[tid].sort(key=lambda x: x.get('ts', 0))

    def get_stack_at_ts(tid, target_ts):
        stack = []
        for op in cpu_ops.get(tid, []):
            start = op.get('ts', 0)
            end = start + op.get('dur', 0)
            if start <= target_ts <= end:
                stack.append(op)
            if start > target_ts:
                break
        return stack

    totals_by_caller = defaultdict(float)
    counts_by_caller = defaultdict(int)
    example_stacks = {}

    for gc in gpu_copies:
        corr = gc['correlation']
        dur = gc['dur']
        launch = launch_events.get(corr)
        if launch:
            tid = launch.get('tid')
            ts = launch.get('ts')
            stack = get_stack_at_ts(tid, ts)
            
            # The innermost op is the last one in the stack
            # Look for an operator that has 'CpuStack' or 'Python Stack' or just use its name
            caller_key = "Unknown"
            py_stack = None
            for op in reversed(stack):
                args = op.get('args', {})
                if 'CpuStack' in args:
                    py_stack = args['CpuStack']
                name = op.get('name', '')
                if name.startswith('aten::') or name.startswith('autograd::') or name == 'python_function':
                    caller_key = name
                    break
                    
            if py_stack and len(py_stack) > 0:
                for frame in py_stack:
                    if 'torch' not in frame and 'site-packages' not in frame:
                        caller_key = f"{caller_key} @ {frame}"
                        break
                        
            totals_by_caller[caller_key] += dur
            counts_by_caller[caller_key] += 1
            if caller_key not in example_stacks:
                example_stacks[caller_key] = py_stack
                
    print(f"Aggregated Callers for direct_copy (total {sum(c['dur'] for c in gpu_copies)/1000:.2f} ms):")
    sorted_callers = sorted(totals_by_caller.items(), key=lambda x: x[1], reverse=True)
    
    for caller, total_dur in sorted_callers:
        print(f"\n{total_dur/1000:8.2f} ms ({counts_by_caller[caller]:5d} calls) : {caller}")
        stack = example_stacks[caller]
        if stack:
            print("  Example Stack:")
            for s in stack[:6]:
                print(f"    {s.strip()}")

if __name__ == "__main__":
    main()
