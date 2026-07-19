import os
import sys
import subprocess

def run_script(script_name, cwd=None):
    print(f"--- Executing {script_name} ---")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)

if __name__ == '__main__':
    print("Starting full pipeline orchestration...")
    run_script("01_data_pipeline.py")
    run_script("02_baseline_model.py")
    run_script("03_ml_model_experiment.py")
    run_script("04_generate_playbook.py")
    run_script("01_plot_model_comparison.py", cwd="../figures/gen_scripts")
    run_script("02_plot_triage_results.py", cwd="../figures/gen_scripts")
    print("Pipeline finished successfully. All outputs are strictly in work/outputs/.")
