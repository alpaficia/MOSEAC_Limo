import psutil
import subprocess
import time

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    try:
        result = subprocess.run(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1)
        output = result.stdout
        gpu_usage = parse_gpu_usage(output)
        return gpu_usage
    except subprocess.TimeoutExpired:
        return None

def parse_gpu_usage(tegrastats_output):
    lines = tegrastats_output.splitlines()
    for line in lines:
        if 'GR3D_FREQ' in line:
            parts = line.split()
            for part in parts:
                if part.startswith('GR3D_FREQ'):
                    usage = part.split('@')[1]
                    return float(usage.replace('%', ''))
    return None

def monitor_usage(duration=30, interval=5):
    with open('PATH_TO_SAVE/usage_data.csv', 'w') as file:
        file.write('Timestamp,CPU Usage (%),GPU Usage (%)\n')
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_usage = get_cpu_usage()
            gpu_usage = get_gpu_usage()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            file.write(f'{timestamp},{cpu_usage},{gpu_usage}\n')
            time.sleep(interval)

if __name__ == '__main__':
    monitor_usage()
