import matplotlib.pyplot as plt


def plot_histogram(data, bins=20):
    plt.hist(data, bins=20)
    plt.xlabel('chunk length')
    plt.ylabel('Frequency')
    plt.show()
