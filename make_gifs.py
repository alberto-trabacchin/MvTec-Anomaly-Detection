from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# Setting up paths and classes
images_path = Path("./outputs")
classes = [str(c.stem) for c in images_path.iterdir()]
epochs = np.linspace(start=1, stop=200).astype(int)  # Generates 200 epochs from 1 to 200
my_classes = ['bottle', 'zipper', 'capsule', 'transistor']

images = {c: list(images_path.joinpath(c).iterdir()) for c in classes}
for c in classes:
    images[c] = [list(images_path.joinpath(f"{c}/{e}").iterdir())[0] for e in epochs]

# Function to create GIF with epoch numbers
def create_gif(image_paths, output_path, duration=0.5, resolution=(128, 128)):
    images = []
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        img = img.resize(resolution, Image.LANCZOS)
        
        # Add epoch number on a black box
        draw = ImageDraw.Draw(img)
        font_size = 40
        try:
            font = ImageFont.truetype("fonts/arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        text = f"Epoch {epochs[i]}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        # Calculate the text position for top right alignment
        text_position = (img.width - text_width - 10, 10)
        
        # Drawing a black rectangle
        box_position = (text_position[0] - 5, text_position[1] - 5, 
                        text_position[0] + text_width + 5, text_position[1] + text_height + 5)
        draw.rectangle(box_position, fill="black")
        
        # Adding white text
        draw.text(text_position, text, fill="white", font=font)
        
        images.append(img)
    
    imageio.mimsave(output_path, images, duration=duration, loop=0)

# Creating GIFs for each class
for c in classes:
    image_paths = images[c]
    Path("gifs/").mkdir(exist_ok=True)
    create_gif(image_paths, f"gifs/{c}.gif", duration=1, resolution=(512, 512))
