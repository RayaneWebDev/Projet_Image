import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.main import process_image

if __name__ == "__main__":
    demo_dir = "demo_images_val"
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        os.path.join(demo_dir, f)
        for f in os.listdir(demo_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    for img in images:
        print(f"\n--- {os.path.basename(img)} ---")
        process_image(img, debug=True)
