import matplotlib.pyplot as plt

def plot_convergence(history, algorithm_name):
    """Plot the convergence of the objective function."""
    plt.figure()
    plt.plot(history, label=f'{algorithm_name} Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    plt.title(f'{algorithm_name} Convergence')
    plt.legend()
    plt.grid()
    plt.show()

def compare_algorithms(results):
    """Compare algorithms based on their performance metrics."""
    algorithms = results['Algorithm']
    clique_sizes = results['Clique Size']
    runtimes = results['Time (s)']

    plt.figure(figsize=(10, 5))

    # Bar plot for clique sizes
    plt.subplot(1, 2, 1)
    plt.bar(algorithms, clique_sizes, color='skyblue')
    plt.xlabel('Algorithm')
    plt.ylabel('Clique Size')
    plt.title('Clique Size Comparison')

    # Bar plot for runtimes
    plt.subplot(1, 2, 2)
    plt.bar(algorithms, runtimes, color='salmon')
    plt.xlabel('Algorithm')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Comparison')

    plt.tight_layout()
    plt.show()
