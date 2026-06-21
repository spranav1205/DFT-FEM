import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Load the data from the CSV file
    data = pd.read_csv("energy_convergence.csv")

    # Extract the columns
    num_elements = data["total_cells"]
    total_charge = data["total_charge"]
    energy = data["energy"]

    # Plot total charge vs number of elements
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(num_elements, total_charge, marker='o')
    # plt.xlabel("Number of Elements")
    # plt.ylabel("Total Charge")
    # plt.title("Total Charge vs Number of Elements")
    # plt.grid()

    # Plot energy vs number of elements
    # plt.subplot(1, 2, 2)
    plt.plot(num_elements, energy, marker='o')
    plt.xlabel("Number of Elements")
    plt.ylabel("Energy")
    plt.title("Energy vs Number of Elements")
    plt.grid()    

    # Show the plots
    plt.tight_layout()
    plt.show()