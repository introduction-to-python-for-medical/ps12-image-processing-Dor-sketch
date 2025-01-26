from image_utils import load_image, edge_detection

image = load_image('image.png')
edges = edge_detection(image)
print(edges)