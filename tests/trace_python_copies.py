import gzip
import json
import sys
from collections import defaultdict


def summarize_trace_copy_callers(trace_file, min_ms=0.01, top_k=20):
    with gzip.open(trace_file, "rt") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    tid_events = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat", "") in (
            "cpu_op",
            "python_function",
            "user_annotation",
        ):
            tid_events[event["tid"]].append(event)

    for tid in tid_events:
        tid_events[tid].sort(key=lambda x: x["ts"])

    copy_durations = defaultdict(float)
    copy_counts = defaultdict(int)

    for _, events_for_tid in tid_events.items():
        active_python = []
        for event in events_for_tid:
            ts = event["ts"]
            dur = event.get("dur", 0)
            name = event["name"]

            active_python = [
                frame for frame in active_python if frame["ts"] + frame.get("dur", 0) > ts
            ]

            if name == "python_function":
                stack_frame = event.get("args", {}).get("name", "Unknown")
                active_python.append({"ts": ts, "dur": dur, "frame": stack_frame})
            elif name in ("aten::copy_", "aten::to"):
                caller = active_python[-1]["frame"] if active_python else "Unknown"
                dur_ms = dur / 1000.0
                if dur_ms > min_ms:
                    copy_durations[caller] += dur_ms
                    copy_counts[caller] += 1

    rows = [
        {
            "caller": caller,
            "duration_ms": duration_ms,
            "count": copy_counts[caller],
        }
        for caller, duration_ms in sorted(
            copy_durations.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
    ]
    return rows

def main():
    trace_file = sys.argv[1]

    rows = summarize_trace_copy_callers(trace_file)

    print("Aggregate PyTorch Copies mapped to Python functions:")
    for row in rows:
        print(f"{row['duration_ms']:8.2f} ms ({row['count']:4d} calls) : {row['caller']}")

if __name__ == '__main__':
    main()
