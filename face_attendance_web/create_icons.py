from PIL import Image, ImageDraw
import os

# Create icons folder
os.makedirs('static/icons', exist_ok=True)

# Create simple green icons
for size in [192, 512]:
    img = Image.new('RGB', (size, size), color='#00ff88')
    img.save(f'static/icons/icon-{size}.png')

print("✅ Icons created!")