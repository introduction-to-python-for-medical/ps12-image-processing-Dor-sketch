from image_utils import load_image, edge_detection, preprocess_image

image = load_image('image.png')
image = preprocess_image(image)
edges = edge_detection(image)
print(edges)