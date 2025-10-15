from app.models.detector import ObjectDetector
import os

# Create detector
print("Loading detector...")
detector = ObjectDetector()

# Get model info
info = detector.get_model_info()
print(f"\nModel Info:")
print(f"Type: {info['model_type']}")
print(f"Device: {info['device']}")
print(f"Number of classes: {len(info['classes'])}")
print(f"Sample classes: {info['classes'][:10]}")

# Test with a sample image (you'll need to provide one)
print("\n" + "="*50)
print("To test detection, place an image in the project folder")
print("and uncomment the code below, replacing 'test.jpg' with your image name")
print("="*50)

# Uncomment these lines when you have a test image:
image_path = r"C:\Users\khale\Downloads\bus_train.jpg"
if os.path.exists(image_path):
    print(f"\nDetecting objects in {image_path}...")
    detections = detector.detect_objects(image_path)
    
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class']}: {det['confidence']:.2f}")
    
    # Save annotated image
    output_path = "test_output.jpg"
    detector.annotate_image(image_path, output_path)
    print(f"\nAnnotated image saved to: {output_path}")
else:
    print(f"\nImage not found: {image_path}")