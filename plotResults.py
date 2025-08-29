import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

plt.figure()
plt.plot(df["N"], df["CPU_time"], marker="o", label="CPU")
plt.plot(df["N"], df["GPU_time"], marker="o", label="GPU")
plt.xlabel("Matrix size N")
plt.ylabel("Time (ms)")
plt.title("CPU vs GPU Matrix Multiplication")
plt.legend()
plt.grid(True)
plt.savefig("runtime.png")

plt.figure()
speedup = df["CPU_time"] / df["GPU_time"]
plt.plot(df["N"], speedup, marker="o", color="red")
plt.xlabel("Matrix size N")
plt.ylabel("Speedup (CPU_time / GPU_time)")
plt.title("GPU Speedup over CPU")
plt.grid(True)
plt.savefig("speedup.png")

plt.show()
