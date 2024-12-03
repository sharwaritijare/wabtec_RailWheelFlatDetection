import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load the image
image_path = "C:/Users/mailv/OneDrive/Pictures/Saved Pictures/test3.png"  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Step 2: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Edge detection using Canny
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # Step 5: Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Draw contours on the original image for visualization
    contour_image = image.copy()

    # Step 7: Analyze dimensions of detected contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Only process contours with area >= 10 pixels
        if area >=20:
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the bounding box on the original image
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Print dimensions and area
            print(f'Contour Area: {area:.2f}, Width: {w}, Height: {h}')

    # Step 8: Display results using Matplotlib for compatibility with Jupyter notebooks
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Contours (Area >= 10)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
