import os
import re
import subprocess
import threading
import queue
import psutil

NUM_GPUS = 8
JOBS_PER_GPU = 1
CPUS_PER_JOB = 24
SCRIPT = "uv run tests/quantization/test_mxfp_training_all_experiments.py"

# === Step 1: Discover config files ===
def get_config_ids(folder):
    files = os.listdir(folder)
    ids = []
    for f in files:
        match = re.match(r'config_(\d+)\.pkl', f)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)

# === Step 2: Define Job and Slot ===
def make_job_commands(folder, config_ids):
    return [f"{SCRIPT} --file_config {folder} --config_id {cid}" for cid in config_ids]

def assign_slots():
    slots = []
    for gpu_id in range(NUM_GPUS):
        for job_in_gpu in range(JOBS_PER_GPU):
            global_slot_id = gpu_id * JOBS_PER_GPU + job_in_gpu
            cpu_start = global_slot_id * CPUS_PER_JOB
            cpus = list(range(cpu_start, cpu_start + CPUS_PER_JOB))
            slots.append({'gpu': gpu_id, 'cpus': cpus})
    return slots

# === Step 3: Worker ===
def run_job(job_cmd, slot):
    import datetime

    # Extract config_id for naming the log file
    match = re.search(r'--config_id\s+(\d+)', job_cmd)
    config_id = match.group(1) if match else "unknown"

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"job_{config_id}.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(slot['gpu'])

    print(f"[{datetime.datetime.now()}] Launching config {config_id} on GPU {slot['gpu']} CPUs {slot['cpus']}")

    with open(log_file, "w") as log_f:
        log_f.write(f"Launching: {job_cmd} on GPU {slot['gpu']} CPUs {slot['cpus']}\n")
        log_f.flush()
        p = subprocess.Popen(
            job_cmd,
            shell=True,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: psutil.Process().cpu_affinity(slot['cpus'])
        )
        p.wait()

    print(f"[{datetime.datetime.now()}] Finished config {config_id}")

def worker(job_q, slot_q):
    while not job_q.empty():
        try:
            slot = slot_q.get(timeout=1)
            job_cmd = job_q.get_nowait()
            run_job(job_cmd, slot)
            job_q.task_done()
            slot_q.put(slot)
        except queue.Empty:
            break

def main(folder):
    config_ids = get_config_ids(folder)
    job_cmds = make_job_commands(folder, config_ids)

    job_q = queue.Queue()
    for cmd in job_cmds:
        job_q.put(cmd)

    slot_q = queue.Queue()
    for slot in assign_slots():
        slot_q.put(slot)

    threads = []
    for _ in range(len(assign_slots())):
        t = threading.Thread(target=worker, args=(job_q, slot_q))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python launcher.py FOLDER_NAME")
        exit(1)
    main(sys.argv[1])
