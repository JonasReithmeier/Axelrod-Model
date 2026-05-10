import subprocess
import sys
import time
from datetime import datetime, timedelta

# List your runner files here in the order you want them to run.
TASKS = [
    "task1_runner.py",        # Task 1: Regular Lattice
    "task2_runner.py",     # Task 2: Small World
    "task3_runner.py"      # Task 3: Axelrod-Schelling
]

def print_header(text):
    print("\n" + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}")
    print("="*60 + "\n")

def main():
    total_start_time = time.perf_counter()
    print_header("STARTING OVERNIGHT SCHEDULER")

    for script in TASKS:
        print_header(f"LAUNCHING: {script}")
        task_start = time.perf_counter()
        
        try:
            # sys.executable ensures the script uses the exact same Python 
            # environment/virtualenv that the scheduler is using.
            result = subprocess.run([sys.executable, script], check=False)
            
            # Check if the process was killed or failed
            if result.returncode != 0:
                print(f"\n[WARNING] {script} exited with error code {result.returncode}.")
                print("Scheduler will continue to the next task anyway to preserve the night.")
                
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Could not launch {script}. Error: {e}")
            
        task_duration = time.perf_counter() - task_start
        print(f"\n>>> FINISHED {script} in {timedelta(seconds=int(task_duration))}\n")

    # Final wrap-up
    total_duration = time.perf_counter() - total_start_time
    print_header(f"ALL SCHEDULED TASKS COMPLETED")
    print(f"Total Uptime: {timedelta(seconds=int(total_duration))}")
    print("You can safely close this terminal.")

if __name__ == "__main__":
    main()