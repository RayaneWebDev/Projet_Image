import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.main_kmeans import process_image_kmeans

if __name__ == "__main__":
    demo_dir = "demo/demo_images_val"
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.isdir(demo_dir):
        print(f"Dossier '{demo_dir}' introuvable.")
    else:
        images = sorted([
            os.path.join(demo_dir, f)
            for f in os.listdir(demo_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])
        for img in images:
            print(f"\n--- {os.path.basename(img)} ---")
            process_image_kmeans(img, debug=True)
