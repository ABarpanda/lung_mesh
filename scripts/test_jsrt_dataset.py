'''
154 JPCLN + 93 JPCNN
'''
import matplotlib.pyplot as plt
from PIL import Image

def plot_nodule_point(study_id, x, y):
    """
    Loads an X-ray image corresponding to the study_id and plots a marker at (x, y).
    
    Parameters:
    - study_id (str): The filename of the image (e.g., 'JPCLN001.png')
    - x (int/float): The x-coordinate of the point
    - y (int/float): The y-coordinate of the point
    - image_dir (str): Directory where the images are stored
    """
    image_path = f"dataset/JSRT_images/{study_id}"
    
    try:
        # Load the image
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image not found at {image_path}. Please check your directory path.")
        return

    # Set up the matplotlib plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    
    # Plot the point at (x, y)
    ax.scatter(x, y, color='red', s=120, marker='x', linewidths=2, label=f'Point ({x}, {y})')
    
    # Formatting
    ax.set_title(f"Study ID: {study_id} | Coordinates: ({x}, {y})", fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with pandas to iterate through your dataset:
import pandas as pd
df = pd.read_csv('dataset/jsrt_metadata.csv')
for index, row in df.iterrows():
    plot_nodule_point(row['study_id'], row['x'], row['y'])
