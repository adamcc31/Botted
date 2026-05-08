import subprocess
import os

os.makedirs('dataset/clob_log', exist_ok=True)

# Get list of files
print("Fetching list of clob_log files from Railway...")
result = subprocess.run('railway ssh "find /app/data/exports -name \'clob_log*.csv\'"', shell=True, capture_output=True, text=True)
files = [line.strip() for line in result.stdout.split('\n') if line.strip() and 'clob_log' in line]
print(f"Found {len(files)} files.")

for f in files:
    # f looks like: /app/data/exports/2026-05-03_132049/clob_log.csv
    parts = f.split('/')
    if len(parts) >= 2:
        dir_name = parts[-2]
        base_name = parts[-1]
        local_name = f"dataset/clob_log/{dir_name}_{base_name}"
        
        print(f"Extracting {f} to {local_name}...")
        with open(local_name, 'w', encoding='utf-8') as out_f:
            subprocess.run(f'railway ssh "cat {f}"', shell=True, stdout=out_f, text=True)

print("Done extracting.")
