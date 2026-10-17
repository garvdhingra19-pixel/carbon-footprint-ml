import time
import platform
import psutil
from codecarbon import EmissionsTracker

def get_system_metadata():
    """Captures fixed hardware context for fair audit comparisons."""
    info = {
        "OS": platform.system() + " " + platform.release(),
        "CPU": platform.processor(),
        "Logical Cores": psutil.cpu_count(logical=True),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }
    return info

def measure_idle_baseline(duration_seconds=30):
    """Measures background power draw while machine is at rest."""
    print(f"Profiling idle baseline for {duration_seconds} seconds (please do not use system)...")
    tracker = EmissionsTracker(
        project_name="idle_baseline",
        output_dir="output",
        log_level="error",
        measure_power_secs=1
    )
    tracker.start()
    time.sleep(duration_seconds)
    emissions_kg = tracker.stop()
    return emissions_kg * 1000  # returns grams CO2e

if __name__ == "__main__":
    print("--- System Telemetry ---")
    for k, v in get_system_metadata().items():
        print(f"{k}: {v}")
    
    idle_co2_g = measure_idle_baseline(15)
    print(f"Idle Baseline (15s): {idle_co2_g:.6f} g CO2e")