import matplotlib.pyplot as plt

def plot_and_save(test_losses, train_losses, test_accs, train_accs, plot_name):
    fig, axs = plt.subplots(2)
    # Plotting the data for the first subplot
    axs[0].plot(range(len(train_losses)), train_losses, label="train loss")
    axs[0].set_title('Loss comparison')
    axs[0].plot(range(len(test_losses)), test_losses, label="test loss")
    axs[0].legend()

    axs[1].plot(range(len(train_accs)), train_accs, label="train acc")
    axs[1].set_title('Accuracy comparison')
    axs[1].plot(range(len(test_accs)), test_accs, label="test acc")
    axs[1].legend()

    # Adding labels and title to the overall plot
    fig.suptitle('Train vs test Comparison')
    fig.text(0.5, 0.04, 'Index', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')

    # Adjusting the layout and spacing
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.4, hspace=0.4)
    plt.savefig(plot_name + ".png")