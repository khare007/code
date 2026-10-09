import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
# Load the dataset (handles European format: semi-colon separated, comma for decimals)
df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",")

# Drop completely empty rows and columns
df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

# The dataset uses -200 to represent missing values; replace them with NaN
df.replace(-200, np.nan, inplace=True)

# Forward-fill and backward-fill missing data to ensure continuous visualizations
df = df.ffill().bfill()

# Take a subset of 100 hours for clearer visual demonstrations
df_sample = df.head(100).copy()


# ==========================================
# 1. BRIGHTNESS (Color Intensity Mapping)
# ==========================================
plt.figure(figsize=(8, 5))
# Mapping Temperature (T) to the 'c' (color) parameter using a sequential colormap
scatter = plt.scatter(df_sample['CO(GT)'], df_sample['NOx(GT)'], 
                      c=df_sample['T'], cmap='hot', edgecolor='k', alpha=0.8)

plt.colorbar(scatter, label='Temperature (°C)')
plt.title("Brightness: CO vs NOx (Color Brightness = Temperature)")
plt.xlabel("CO(GT) Concentration")
plt.ylabel("NOx(GT) Concentration")
plt.tight_layout()
# plt.savefig("1_brightness.png") # Uncomment to save for lab file
plt.show()


# ==========================================
# 2. ORIENTATION (Vector Field / Trajectory)
# ==========================================
plt.figure(figsize=(8, 5))
T = df_sample['T'].values
RH = df_sample['RH'].values

# Calculate directional vectors pointing to the next hour's environmental state
u = np.diff(T)
v = np.diff(RH)

# Quiver plot maps orientation (angle and magnitude) of the state change
plt.quiver(T[:-1], RH[:-1], u, v, color='blue', angles='xy', scale_units='xy', scale=1, alpha=0.6)
plt.title("Orientation: Hourly Trajectory of Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Relative Humidity (%)")
plt.tight_layout()
# plt.savefig("2_orientation.png")
plt.show()


# ==========================================
# 3. TEXTURE (Hatch Patterns)
# ==========================================
plt.figure(figsize=(8, 5))
# Calculate means (CO is scaled x10 so it remains visible next to NOx)
means = [df_sample['CO(GT)'].mean() * 10, 
         df_sample['NO2(GT)'].mean(), 
         df_sample['NOx(GT)'].mean()]

labels = ['CO (Scaled x10)', 'NO2', 'NOx']
bars = plt.bar(labels, means, color='lightgray', edgecolor='black', linewidth=1.5)

# Applying distinct textures (hatches) to each individual bar
hatches = ['///', '\\\\\\', 'xxx']
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

plt.title("Texture: Average Pollutant Levels with Distinct Hatches")
plt.ylabel("Concentration")
plt.tight_layout()
# plt.savefig("3_texture.png")
plt.show()


# ==========================================
# 4. MOTION (Time-Series Animation)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 50)
ax.set_ylim(0, df_sample['CO(GT)'].head(50).max() + 1)
ax.set_title("Motion: CO(GT) Concentration Evolving Over Time")
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("CO(GT) Concentration")

line, = ax.plot([], [], lw=2, color='red', marker='o')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    # Update the data mapped to the line for each frame
    x = np.arange(i + 1)
    y = df_sample['CO(GT)'].iloc[:i + 1]
    line.set_data(x, y)
    return line,

# Generate the animation over 50 frames
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=50, interval=100, blit=True)

# NOTE FOR LAB FILE: To save this animation as a GIF, you must have the 'pillow' library installed.
# Uncomment the line below to save the motion output to your folder:
# ani.save("4_motion.gif", writer='pillow')

plt.show()
