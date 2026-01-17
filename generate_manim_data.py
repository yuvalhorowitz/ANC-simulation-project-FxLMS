import numpy as np
import pyroomacoustics as pra
from scipy.signal import chirp

# ==========================================
# CONFIGURATION
# ==========================================
FS = 4000
DURATION = 6.0 
N_SAMPLES = int(FS * DURATION)
L_FILTER = 64   # Filter length
MU = 0.005      # Step size
NUM_SPEAKERS = 4
CAR_DIMS = [2.5, 1.5, 1.2]

# ==========================================
# 1. PHYSICS SETUP (4 Speakers + Engine)
# ==========================================
print("1. Setting up Car Acoustics...")
room = pra.ShoeBox(CAR_DIMS, fs=FS, max_order=12, absorption=0.6)

# Positions
mic_pos = [0.5, 0.4, 1.0]      # Driver
eng_pos = [2.3, 0.75, 0.5]     # Engine
# 4 Corner Speakers
spk_pos = [
    [1.8, 0.3, 0.8], # FL
    [1.8, 1.2, 0.8], # FR
    [0.4, 0.3, 0.8], # RL
    [0.4, 1.2, 0.8]  # RR
]

room.add_source(eng_pos)
for pos in spk_pos: room.add_source(pos)
room.add_microphone(mic_pos)
room.compute_rir()

# Extract Impulse Responses
hp_ir = room.rir[0][0][:256] 
hp_ir /= np.sqrt(np.sum(hp_ir**2)) # Normalize

hs_irs = []
for i in range(NUM_SPEAKERS):
    h = room.rir[0][i+1][:256]
    h /= (np.sqrt(np.sum(h**2)) + 1e-10)
    hs_irs.append(h)

# ==========================================
# 2. SIGNAL GENERATION (Acceleration)
# ==========================================
print("2. Generating Engine Acceleration Noise...")
t = np.arange(N_SAMPLES) / FS
# Chirp from 80Hz to 200Hz (Revving engine)
ref_noise = chirp(t, f0=80, f1=200, t1=DURATION, method='linear') 
# Add some random road noise
ref_noise += 0.05 * np.random.normal(0, 1, N_SAMPLES)
ref_noise = ref_noise / np.std(ref_noise)

# ==========================================
# 3. FxLMS ALGORITHM (Data Recording)
# ==========================================
print("3. Running FxLMS & Recording Data...")

W = np.zeros((NUM_SPEAKERS, L_FILTER))
x_buffer = np.zeros(L_FILTER)
fx_buffers = np.zeros((NUM_SPEAKERS, 256))
x_filtered_buffers = np.zeros((NUM_SPEAKERS, L_FILTER))
buffer_hp = np.zeros(len(hp_ir))
buffers_hs = np.zeros((NUM_SPEAKERS, 256))

# DATA CONTAINERS FOR VIDEO
log_error = np.zeros(N_SAMPLES)
log_weights = [] 

for n in range(N_SAMPLES):
    x_n = ref_noise[n]
    
    # Update buffers
    x_buffer = np.roll(x_buffer, 1)
    x_buffer[0] = x_n
    
    # Calculate Outputs
    y_signals = [np.dot(W[k], x_buffer) for k in range(NUM_SPEAKERS)]
    
    # Physics (Primary)
    buffer_hp = np.roll(buffer_hp, 1)
    buffer_hp[0] = x_n
    d_n = np.dot(buffer_hp, hp_ir)
    
    # Physics (Secondary Sum)
    s_total = 0
    for k in range(NUM_SPEAKERS):
        buffers_hs[k] = np.roll(buffers_hs[k], 1)
        buffers_hs[k][0] = y_signals[k]
        s_total += np.dot(buffers_hs[k], hs_irs[k])
        
    e_n = d_n + s_total
    log_error[n] = e_n # RECORD ERROR
    
    # Update Weights
    for k in range(NUM_SPEAKERS):
        fx_buffers[k] = np.roll(fx_buffers[k], 1)
        fx_buffers[k][0] = x_n
        x_prime = np.dot(fx_buffers[k], hs_irs[k])
        
        x_filtered_buffers[k] = np.roll(x_filtered_buffers[k], 1)
        x_filtered_buffers[k][0] = x_prime
        
        pwr = np.dot(x_filtered_buffers[k], x_filtered_buffers[k]) + 1e-6
        W[k] = W[k] * 0.9999 - (MU / pwr) * e_n * x_filtered_buffers[k]

    # Snapshot weights every 50 samples (4000/50 = 80 fps for video)
    if n % 50 == 0:
        log_weights.append(W.copy())

# ==========================================
# 4. EXPORT
# ==========================================
print("4. Saving to 'anc_data.npz'...")
np.savez('anc_data.npz', 
         error=log_error, 
         weights=np.array(log_weights))
print("Done! You can now run the Manim renderer.")