import matplotlib.pyplot as plt

def plot(N_EPOCHS, g_loss, d_loss):
    epochs = [i for i in range(N_EPOCHS)]

    plt.figure(1)
    plt.plot(epochs, g_loss)
    plt.xlabel('epochs')
    plt.ylabel('Generator loss')
    plt.legend(["Generator loss"])
    plt.savefig('plots/generator_loss_plot.png')

    plt.figure(2)
    plt.plot(epochs, d_loss,'r')
    plt.xlabel('epochs')
    plt.ylabel('Discriminator loss')
    plt.legend(["Discriminator loss"])
    plt.savefig('plots/discriminator_loss_plot.png')
