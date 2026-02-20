#!/usr/bin/env python3
"""
Generate favicon and header icon variants from a source image.

Usage:
    python generate_icons.py <source_image_path>

The script will create resized versions at the appropriate sizes.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow")
    sys.exit(1)


def generate_icons(source_path: Path) -> None:
    """Generate all icon variants from a source image."""
    
    if not source_path.exists():
        print(f"ERROR: Source image not found: {source_path}")
        sys.exit(1)
    
    if not source_path.is_file():
        print(f"ERROR: Source is not a file: {source_path}")
        sys.exit(1)
    
    # Open the source image
    try:
        img = Image.open(source_path)
        print(f"✓ Loaded source image: {source_path.name} ({img.size[0]}x{img.size[1]})")
    except Exception as e:
        print(f"ERROR: Failed to open image: {e}")
        sys.exit(1)
    
    # Ensure the image has an alpha channel for transparency
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Define output sizes
    # (filename, width, height, directory)
    icon_specs = [
        # Favicon sizes
        ("favicon-16x16.png", 16, 16, "icons"),
        ("favicon-32x32.png", 32, 32, "icons"),
        ("favicon-48x48.png", 48, 48, "icons"),
        
        # Header icon
        ("icon-header-48x48.png", 48, 48, "icons"),
        ("icon-header-64x64.png", 64, 64, "icons"),
        
        # Apple touch icon
        ("apple-touch-icon.png", 180, 180, "icons"),
        
        # Android icons
        ("android-chrome-192x192.png", 192, 192, "icons"),
        ("android-chrome-512x512.png", 512, 512, "icons"),
        ("android-chrome-maskable-192x192.png", 192, 192, "icons"),
        ("android-chrome-maskable-512x512.png", 512, 512, "icons"),
        
        # MS tile
        ("mstile-150x150.png", 150, 150, "icons"),
    ]
    
    # Determine base directory (relative to this script)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "src" / "stemmacodicum" / "web" / "static"
    
    if not base_dir.exists():
        print(f"ERROR: Static directory not found: {base_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    icons_dir = base_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {icons_dir}")
    print("\nGenerating icons:\n")
    
    generated = []
    for filename, width, height, subdirectory in icon_specs:
        output_dir = base_dir / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        # Resize image (using LANCZOS for high-quality downsampling)
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Save the resized image
        resized.save(output_path, "PNG", optimize=True)
        generated.append((filename, width, height, output_path))
        print(f"  ✓ {filename:<35} {width:>3}x{height:<3}")
    
    print(f"\n✅ Generated {len(generated)} icon variants")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_icons.py <source_image_path>")
        print("\nExample:")
        print("  python generate_icons.py /path/to/icon.png")
        print("  python generate_icons.py ~/Downloads/stemma-icon.png")
        sys.exit(1)
    
    source_file = Path(sys.argv[1]).expanduser().resolve()
    generate_icons(source_file)
