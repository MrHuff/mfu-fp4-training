import sys
import json
import gzip

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_copies.py <trace_file.json.gz>")
        sys.exit(1)
        
    trace_file = sys.argv[1]
    
    with gzip.open(trace_file, 'rt') as f:
        data = json.load(f)
        
    events = data.get('traceEvents', [])
    
    prints = 0
    for e in events:
        name = e.get('name', '')
        if 'direct_copy' in name and e.get('ph') == 'X':
            print(json.dumps(e, indent=2))
            prints += 1
            if prints >= 3:
                break
                
    # Also find a cpu_op that has a correlation matching one of these
if __name__ == '__main__':
    main()
