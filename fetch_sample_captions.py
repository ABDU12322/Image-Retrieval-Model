import json
import random
from pathlib import Path


def fetch_sample_captions(input_file, output_file, sample_size=30000):
    """
    Fetch a sample of image captions for training a small-scale model.
    
    Args:
        input_file: Path to the cleaned captions JSON file
        output_file: Path to save the sampled captions JSON file
        sample_size: Number of images to sample (default: 2000)
    """
    print(f"Fetching sample captions from: {input_file}")
    print(f"Sample size: {sample_size}")
    
    # Load the cleaned captions
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_annotations = len(data['annotations'])
    print(f"Total available annotations: {total_annotations}")
    
    # Randomly sample the specified number of annotations
    if sample_size > total_annotations:
        print(f"⚠ Warning: Sample size ({sample_size}) is larger than available annotations ({total_annotations})")
        print(f"  Using all {total_annotations} annotations instead")
        sampled_annotations = data['annotations']
    else:
        sampled_annotations = random.sample(data['annotations'], sample_size)
    
    # Create output structure
    output_data = {
        'annotations': sampled_annotations
    }
    
    # Save sampled captions
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Sampled {len(sampled_annotations)} image captions")
    print(f"✓ Saved to: {output_file}\n")


def main():
    # Define paths
    input_file = Path(r'dataset/coco/annotations/cleaned/captions_train2017.json')
    output_dir = Path(r'dataset/coco/annotations/cleaned/')
    output_file = output_dir / 'captions_train2017_sample_30000.json'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FETCHING SAMPLE CAPTIONS FOR SMALL-SCALE TRAINING")
    print("=" * 60)
    print()
    
    if input_file.exists():
        try:
            # Set random seed for reproducibility
            random.seed(42)
            
            fetch_sample_captions(str(input_file), str(output_file), sample_size=30000)
            
            print("=" * 60)
            print("SAMPLING COMPLETE!")
            print("=" * 60)
        except Exception as e:
            print(f"✗ Error processing: {str(e)}\n")
    else:
        print(f"✗ File not found: {input_file}\n")


if __name__ == '__main__':
    main()
