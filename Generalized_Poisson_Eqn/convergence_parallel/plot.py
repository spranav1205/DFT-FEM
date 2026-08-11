import matplotlib.pyplot as plt
import pandas as pd

df3 = pd.read_csv(r"./convergence parallel/3_energy_convergence.csv")
df4 = pd.read_csv(r"./convergence parallel/4_energy_convergence.csv")

# plt.plot(df3['total_cells'][1:], df3['energy'][1:], label='3D', color='blue')
# plt.plot(df4['total_cells'][1:], df4['energy'][1:], label='4D', color='red')

# plt.xlabel('Total Cells')
# plt.ylabel('Energy')
# plt.title('Energy Convergence Comparison')
# plt.legend()
# plt.grid()

# plt.show()

plt.plot(df3['total_cells'], df3['cg_iterations'], label='3D', color='blue')
plt.plot(df4['total_cells'], df4['cg_iterations'], label='4D', color='red')

plt.xlabel('Total Cells')
plt.ylabel('CG Iterations')
plt.title('CG Iterations Convergence Comparison')
plt.legend()
plt.grid()

plt.show()