import matplotlib.pyplot as plt
import os

def plot_training_metrics(avg_reward_per_iter, dist_from_origin_iter, save_dir="./plots"):
    """
    Plot training metrics and save to disk.
    
    Args:
        avg_reward_per_iter (list): List of average rewards per iteration
        dist_from_origin_iter (list): List of distances from origin per iteration
        save_dir (str): Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 4))

    # Plot average reward
    plt.subplot(1, 2, 1)
    plt.plot(avg_reward_per_iter, label='Avg Reward (Train)')
    plt.title('Reward Progress (Training)')
    plt.xlabel('Iteration')
    plt.ylabel('Avg Reward')
    plt.legend()

    # Plot distance from origin
    plt.subplot(1, 2, 2)
    plt.plot(dist_from_origin_iter, label='Distance from Origin', color='orange')
    plt.title('Distance from Origin')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close() 