import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_single_job(config_id, config_folder, log_dir, script_path, python_exec):
    log_path = os.path.join(log_dir, f"mxfp_stacked_{config_id}.out")
    temp_log = f"{log_path}.tmp"

    cmd = [
        python_exec,
        script_path,
        "--file_config", config_folder,
        "--config_id", str(config_id)
    ]

    with open(temp_log, "w") as temp_f:
        process = subprocess.Popen(cmd, stdout=temp_f, stderr=subprocess.STDOUT)
        process.wait()

    # Only keep last 50 lines
    with open(temp_log, "r") as temp_f:
        lines = temp_f.readlines()
    with open(log_path, "w") as final_f:
        final_f.writelines(lines[-50:])

    os.remove(temp_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--num_jobs", type=int, required=True)
    parser.add_argument("--stack_size", type=int, required=True)
    parser.add_argument("--config_folder", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--script_path", type=str, default="tests/quantization/test_mxfp_training_all_experiments.py")
    parser.add_argument("--python_exec", type=str, default="/app/.venv/bin/python")

    args = parser.parse_args()

    end_index = min(args.start_index + args.stack_size, args.num_jobs)
    job_ids = range(args.start_index, end_index)

    print(f"Starting jobs: {list(job_ids)}")

    max_workers = min(2, args.stack_size)  # change to 1 for serial; 2–3 for light stacking
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_job, cid, args.config_folder, args.log_dir, args.script_path, args.python_exec)
            for cid in job_ids
        ]
        for future in as_completed(futures):
            future.result()

    print(f"Finished jobs {args.start_index}–{end_index}")
