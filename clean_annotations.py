import json
import os
from pathlib import Path

def clean_annotations(input_file, output_file):
    """
    Clean annotation files to keep only id, image_id, and caption fields.
    For captions files, groups multiple captions per image.
    
    Args:
        input_file: Path to the input annotation JSON file
        output_file: Path to save the cleaned annotation JSON file
    """
    print(f"Processing: {input_file}")
    
    # Load the original annotations
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is a captions file
    is_captions_file = 'captions' in os.path.basename(input_file)
    
    # Extract the required fields for each annotation
    cleaned_annotations = []
    
    if 'annotations' in data:
        if is_captions_file:
            # For captions files, group multiple captions per image
            from collections import defaultdict
            captions_by_image = defaultdict(lambda: {'id': None, 'captions': []})
            
            for annotation in data['annotations']:
                image_id = annotation.get('image_id')
                captions_by_image[image_id]['id'] = image_id
                captions_by_image[image_id]['captions'].append(annotation.get('caption'))
            
            # Convert to list of cleaned entries
            for image_id, entry in captions_by_image.items():
                cleaned_entry = {
                    'id': entry['id'],
                    'captions': entry['captions']
                }
                cleaned_annotations.append(cleaned_entry)
        else:
            # For non-captions files, keep the original structure
            for annotation in data['annotations']:
                cleaned_entry = {
                    'id': annotation.get('id'),
                    'image_id': annotation.get('image_id'),
                    'caption': annotation.get('caption')
                }
                cleaned_annotations.append(cleaned_entry)
    
    # Create output structure with cleaned annotations
    output_data = {
        'annotations': cleaned_annotations
    }
    
    # Save cleaned annotations
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Cleaned {len(cleaned_annotations)} annotations")
    print(f"✓ Saved to: {output_file}\n")


def main():
    # Define paths to annotation files
    annotation_dir = Path(r'dataset/coco/annotations/')
    
    annotation_files = [
        'captions_train2017.json',
        'captions_val2017.json'
    ]
    
    # Create output directory if it doesn't exist
    output_dir = annotation_dir / 'cleaned'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CLEANING ANNOTATION FILES")
    print("=" * 60)
    print(f"Input directory: {annotation_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Process each annotation file
    for filename in annotation_files:
        input_path = annotation_dir / filename
        output_path = output_dir / filename
        
        if input_path.exists():
            try:
                clean_annotations(str(input_path), str(output_path))
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}\n")
        else:
            print(f"⚠ File not found: {input_path}\n")
    
    print("=" * 60)
    print("CLEANING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
