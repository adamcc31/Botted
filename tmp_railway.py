import subprocess
print(subprocess.run(['railway', 'ssh', '-e', 'production', 'find /app -name "*.db" -o -name "*.json" | grep -i "trade\\|state\\|v5"'], capture_output=True, text=True).stdout)
