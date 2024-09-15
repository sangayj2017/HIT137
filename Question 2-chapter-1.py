from PIL import Image
import numpy as np

# Load the image
image = Image.open('chapter1.jpg')
pixels = np.array(image)

# Define the number n (replace with the actual generated number)
n = 10

# Add n to each pixel value (r, g, b)
pixels = np.clip(pixels + n, 0, 255)  # Ensure values stay within the valid range

# Create a new image with the updated pixel values
new_image = Image.fromarray(pixels.astype('uint8'))
new_image.save('chapter1out.png')

# Sum all the red (r) pixel values in the new image
red_sum = np.sum(pixels[:, :, 0])
print("Sum of all red pixel values:", red_sum)