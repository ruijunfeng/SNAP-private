import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def parse_events_and_save_plot(
    event_file_path: str, 
    output_image_path: str,
):
    """Parse TensorBoard event file and save scalar plots to an image file.
    
    Args:
        event_file_path (str): Path to the TensorBoard event file.
        output_image_path (str): Path to save the output image file.
    """
    # 1. Initialize the EventAccumulator.
    # size_guidance is a dictionary that tells the accumulator how much data to load.
    # Setting SCALARS to 0 loads all scalar data.
    ea = event_accumulator.EventAccumulator(
        event_file_path,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    
    # 2. Load the events from the file.
    ea.Reload()
    
    # 3. Get all available scalar tags (e.g., 'Loss/train', 'Accuracy/validation').
    scalar_tags = ea.Tags()['scalars']
    
    # 4. Create subplots.
    # The number of rows will match the number of tags to plot each on its own subplot.
    num_tags = len(scalar_tags)
    fig, axes = plt.subplots(num_tags, 1, figsize=(12, 6 * num_tags), squeeze=False)
    # Using squeeze=False ensures `axes` is always a 2D array, which simplifies indexing.
    axes = axes.flatten() # Flatten the 2D array to 1D for easy iteration.
    
    # 5. Iterate through each tag, extract its data, and plot it.
    for i, tag in enumerate(scalar_tags):
        # Extract scalar events for the current tag.
        scalar_events = ea.Scalars(tag)
        
        # Extract the step and value for each event point.
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        
        # Plot the data on the corresponding subplot.
        ax = axes[i]
        ax.plot(steps, values, label=tag, color=f'C{i}')
        ax.set_title(tag, fontsize=16)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    
    # 6. Adjust layout and save the figure.
    fig.tight_layout(pad=3.0) # Adjust subplot params for a tight layout.
    
    # Save the figure to the specified path.
    # `bbox_inches='tight'` helps prevent labels from being cut off.
    plt.savefig(output_image_path, dpi=150, bbox_inches="tight")
    
    # Close the figure to free up memory.
    plt.close(fig)
