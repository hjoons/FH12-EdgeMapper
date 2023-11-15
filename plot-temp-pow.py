from subprocess import PIPE, Popen
import matplotlib.pyplot as plt
import re

seconds = []
cpu_temps = []
gpu_temps = []
board_pows = []
asic_pows = []
board_pows = []

sudo_password = '12346578'

# CPU temp output: CPU@xx.xxxC
# GPU temp output: GPU@xx.xxxC
# CPU_GPU pow output: VDD_CPU_GPU_CV xxxmW
# Board pow output: VDD_IN xxxmW
CPU_temp_pattern = re.compile(r'CPU@(\d+(\.\d+)?)C')
GPU_temp_pattern = re.compile(r'GPU@(\d+(\.\d+)?)C')
CPU_GPU_pow_pattern = re.compile(r'VDD_CPU_GPU_CV (\d+)mW')
Board_pow_pattern = re.compile(r'VDD_IN (\d+)mW')

command = ['sudo', '-S', 'tegrastats']

# start tegrastats in background
process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)

# login as sudo
process.stdin.write('12345678\n')
process.stdin.flush()

# create plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Create lines for subplot 1
line1_sub1, = ax1.plot([], [], color='blue', label='CPU Temp')  # Blue for Line 1
line2_sub1, = ax1.plot([], [], color='red', label='GPU Temp')  # Red for Line 2
ax1.legend()
ax1.set_ylabel('Temperature (C)')
ax1.grid(True)

# Create lines for subplot 2
line1_sub2, = ax2.plot([], [], color='green', label='CPU + GPU Power')  # Green for Line 1
line2_sub2, = ax2.plot([], [], color='yellow', label='Board Power')  # Yellow for Line 2
ax2.legend()
ax2.set_ylabel('Power Consumption (W)')
ax2.set_xlabel('Time (s)')
ax2.grid(True)

# for x-axis
second = 1

while True:
    data = process.stdout.readline()
    CPU_temp = float(CPU_temp_pattern.search(data).group(1))
    GPU_temp = float(GPU_temp_pattern.search(data).group(1))
    Board_pow = float(Board_pow_pattern.search(data).group(1)) / 1000
    ASIC_pow = float(CPU_GPU_pow_pattern.search(data).group(1)) / 1000
    cpu_temps.append(CPU_temp)
    gpu_temps.append(GPU_temp)
    asic_pows.append(ASIC_pow)
    board_pows.append(Board_pow)
    seconds.append(second)
    second += 1
    #print(f'Temps: {CPU_temp} C CPU, {GPU_temp} C GPU')
    #print(f'Pow: {ASIC_pow} W asic, {Board_pow} W board')
    
    # update plots
    line1_sub1.set_data(seconds, cpu_temps)
    line2_sub1.set_data(seconds, gpu_temps)
    line1_sub2.set_data(seconds, asic_pows)
    line2_sub2.set_data(seconds, board_pows)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    plt.draw()
    plt.pause(0.5)

plt.show()

