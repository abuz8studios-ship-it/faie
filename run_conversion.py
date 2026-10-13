"""Detached launcher for FAIE conversion with progress logging."""
import sys
import os
import time

os.chdir(r"E:\ABU\NEOCORTEX")
sys.path.insert(0, r"E:\ABU\NEOCORTEX")

LOG_FILE = r"E:\ABU\NEOCORTEX\conversion_progress.log"

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'a', buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee(LOG_FILE)
sys.stderr = sys.stdout

with open(LOG_FILE, 'w') as f:
    f.write(f"FAIE Conversion started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"PID: {os.getpid()}\n")
    f.write("=" * 70 + "\n")

from faie_convert import convert_full_model, verify_conversion
import torch
import torch.nn.functional as F

try:
    success = convert_full_model()
    if success:
        print("\n\nConversion complete. Running 7-step verification...\n")
        verify_conversion()
    else:
        print("\nConversion FAILED.")
except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    print(f"\nFinished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
