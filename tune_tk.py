import os
import subprocess
import re
import shlex

SHAPE = "M=65536, K=2048, N=8192"
CU_FILE = "/opt/mfu/EXTERNAL_PATH"

configs_to_test = [
    "<256, 4, 8, 12, 2, false>", # base
    "<256, 5, 8, 12, 2, false>", # larger load pipe
    "<256, 4, 16, 1, 2, false>", # tested 
    "<256, 4, 16, 12, 2, false>", # N=16384 default
    "<256, 5, 8, 4, 2, true>",   # N=2048 default (didn't work great)
    "<256, 5, 4, 12, 2, true>",  # tuned for 2048 width maybe 
    "<256, 4, 4, 12, 2, false>", 
    "<128, 5, 4, 12, 2, true>",  # tiny square
    "<128, 4, 8, 12, 2, true>"
]

def replace_config(new_config):
    with open(CU_FILE, 'r') as f:
        content = f.read()
    
    # We replace specifically the block inside if (K <= 2048) { ... using C = ...
    pattern = r'(if \(K <= 2048\) \{\n\s*using C = nvfp4_gemm::config)<[^>]+>;'
    replacement = fr'\1{new_config};'
    new_content = re.sub(pattern, replacement, content)
    
    with open(CU_FILE, 'w') as f:
        f.write(new_content)

def run_test():
    # Make
    p = subprocess.run(["make", "clean"], cwd="/opt/mfu/EXTERNAL_PATH", capture_output=True)
    p = subprocess.run(["make"], cwd="/opt/mfu/EXTERNAL_PATH", capture_output=True)
    if p.returncode != 0:
        return "BUILD_FAILED"
    
    # Run test
    p = subprocess.run(["python", "debug_shapes.py"], cwd="/opt/mfu/EXTERNAL_PATH", capture_output=True, text=True)
    out = p.stdout
    
    # Parse the specific shape
    for line in out.splitlines():
        if "Benchmarking M=65536, K=2048, N=8192" in line:
            # next line should have the result
            idx = out.find("Benchmarking M=65536, K=2048, N=8192")
            subout = out[idx:]
            for l2 in subout.splitlines():
                if "TK:" in l2:
                    return l2.strip()
    return "NO_TFLOPS_FOUND"

best_config = None
best_tflops = 0

for cfg in configs_to_test:
    print(f"Testing {cfg}...")
    replace_config(cfg)
    res = run_test()
    print(f"  Result: {res}")
    
    if "TFLOPS" in res:
        try:
            val = float(res.split("ms,")[1].split("TFLOPS")[0].strip())
            if val > best_tflops:
                best_tflops = val
                best_config = cfg
        except:
            pass

print(f"\n====================================")
print(f"BEST CONFIG: {best_config} at {best_tflops} TFLOPS")
