"""
Market-1501 Dataset Organizer
==============================
This script organizes a flat folder of Market-1501 images into the proper structure.

The official dataset has these splits based on identity ranges:
- Training: identities with smaller IDs (first 751 IDs)
- Testing/Gallery: identities with larger IDs (remaining 750 IDs)

Since you have a flat structure, we'll reorganize based on the standard protocol.
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def parse_market1501_filename(filename: str) -> Dict:
    """Parse Market-1501 filename to extract metadata."""
    pattern = r'(-?\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.jpg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    person_id = int(match.group(1))
    camera_id = int(match.group(2))
    sequence_id = int(match.group(3))
    frame_id = int(match.group(4))
    detection_idx = int(match.group(5))
    
    return {
        'person_id': person_id,
        'camera_id': camera_id,
        'sequence_id': sequence_id,
        'frame_id': frame_id,
        'detection_idx': detection_idx,
        'is_distractor': person_id == 0,
        'is_junk': person_id == -1
    }


def analyze_folder(folder_path: str) -> Dict:
    """Analyze the contents of a folder with Market-1501 images."""
    folder = Path(folder_path)
    
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'distractors': 0,
        'junk': 0,
        'unparseable': 0,
        'person_ids': set(),
        'camera_ids': set(),
        'images_per_id': defaultdict(int)
    }
    
    for img_file in folder.glob("*.jpg"):
        stats['total_images'] += 1
        info = parse_market1501_filename(img_file.name)
        
        if info is None:
            stats['unparseable'] += 1
            continue
        
        stats['valid_images'] += 1
        
        if info['is_distractor']:
            stats['distractors'] += 1
        elif info['is_junk']:
            stats['junk'] += 1
        else:
            stats['person_ids'].add(info['person_id'])
            stats['images_per_id'][info['person_id']] += 1
        
        stats['camera_ids'].add(info['camera_id'])
    
    return stats


def organize_dataset(source_folder: str, 
                     output_folder: str,
                     train_ratio: float = 0.5,
                     copy_files: bool = True,
                     create_query: bool = True) -> Dict:
    """
    Organize flat Market-1501 images into proper folder structure.
    
    The standard Market-1501 protocol:
    - 751 identities for training
    - 750 identities for testing (gallery + query)
    - Distractors go to gallery (bounding_box_test)
    - Junk images are typically ignored but we'll put them in a separate folder
    
    Args:
        source_folder: Path to folder with all images
        output_folder: Path where organized structure will be created
        train_ratio: Ratio of identities for training (default 0.5 = 50%)
        copy_files: If True, copy files. If False, move files.
        create_query: If True, create query set from test identities
    
    Returns:
        Statistics about the organization
    """
    source = Path(source_folder)
    output = Path(output_folder)
    
    # Create output structure
    train_folder = output / "bounding_box_train"
    test_folder = output / "bounding_box_test"  # Gallery
    query_folder = output / "query"
    junk_folder = output / "junk_images"
    
    for folder in [train_folder, test_folder, query_folder, junk_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing source folder: {source}")
    
    # First pass: collect all images and their metadata
    all_images = []
    for img_file in source.glob("*.jpg"):
        info = parse_market1501_filename(img_file.name)
        if info:
            info['path'] = img_file
            info['filename'] = img_file.name
            all_images.append(info)
    
    print(f"Found {len(all_images)} valid images")
    
    # Get all unique person IDs (excluding distractors and junk)
    valid_ids = sorted(set(
        img['person_id'] for img in all_images 
        if not img['is_distractor'] and not img['is_junk']
    ))
    
    print(f"Found {len(valid_ids)} unique identities")
    
    # Split identities into train and test
    split_point = int(len(valid_ids) * train_ratio)
    train_ids = set(valid_ids[:split_point])
    test_ids = set(valid_ids[split_point:])
    
    print(f"Training identities: {len(train_ids)}")
    print(f"Testing identities: {len(test_ids)}")
    
    # Organize images by identity and camera for query selection
    test_images_by_id_cam = defaultdict(lambda: defaultdict(list))
    
    stats = {
        'train': 0,
        'gallery': 0,
        'query': 0,
        'distractors': 0,
        'junk': 0
    }
    
    operation = shutil.copy2 if copy_files else shutil.move
    op_name = "Copying" if copy_files else "Moving"
    
    print(f"\n{op_name} files...")
    
    for img in all_images:
        src_path = img['path']
        filename = img['filename']
        
        if img['is_junk']:
            # Junk images
            dst_path = junk_folder / filename
            operation(src_path, dst_path)
            stats['junk'] += 1
            
        elif img['is_distractor']:
            # Distractors go to gallery (test folder)
            dst_path = test_folder / filename
            operation(src_path, dst_path)
            stats['distractors'] += 1
            
        elif img['person_id'] in train_ids:
            # Training images
            dst_path = train_folder / filename
            operation(src_path, dst_path)
            stats['train'] += 1
            
        elif img['person_id'] in test_ids:
            # Test images - track for query selection
            test_images_by_id_cam[img['person_id']][img['camera_id']].append(img)
    
    # Create query and gallery from test identities
    if create_query:
        print("Creating query and gallery sets...")
        
        for person_id, cameras in test_images_by_id_cam.items():
            for cam_id, images in cameras.items():
                # Take first image from each camera as query
                query_img = images[0]
                gallery_imgs = images[1:] if len(images) > 1 else []
                
                # Copy query image
                src_path = query_img['path']
                dst_path = query_folder / query_img['filename']
                operation(src_path, dst_path)
                stats['query'] += 1
                
                # Copy remaining to gallery
                for img in gallery_imgs:
                    src_path = img['path']
                    dst_path = test_folder / img['filename']
                    operation(src_path, dst_path)
                    stats['gallery'] += 1
                
                # If only one image per camera, also add to gallery
                if len(images) == 1:
                    dst_path = test_folder / query_img['filename']
                    if not dst_path.exists():
                        operation(query_img['path'], dst_path)
                        stats['gallery'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Organize Market-1501 images into proper folder structure"
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source folder containing all Market-1501 images'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output folder for organized dataset'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze the folder, do not organize'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying (saves disk space)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.5,
        help='Ratio of identities for training (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Market-1501 Dataset Organizer")
    print("=" * 60)
    
    # Analyze first
    print(f"\nAnalyzing: {args.source}")
    stats = analyze_folder(args.source)
    
    print(f"\n📊 Analysis Results:")
    print(f"  Total images: {stats['total_images']:,}")
    print(f"  Valid images: {stats['valid_images']:,}")
    print(f"  Unique identities: {len(stats['person_ids'])}")
    print(f"  Cameras: {sorted(stats['camera_ids'])}")
    print(f"  Distractors (ID=0000): {stats['distractors']:,}")
    print(f"  Junk (ID=-1): {stats['junk']:,}")
    print(f"  Unparseable: {stats['unparseable']}")
    
    if stats['person_ids']:
        print(f"  ID range: {min(stats['person_ids'])} - {max(stats['person_ids'])}")
    
    if args.analyze_only:
        print("\n✓ Analysis complete (--analyze-only mode)")
        return
    
    # Organize
    print(f"\n🔄 Organizing dataset...")
    print(f"  Output: {args.output}")
    print(f"  Mode: {'Move' if args.move else 'Copy'}")
    print(f"  Train ratio: {args.train_ratio}")
    
    org_stats = organize_dataset(
        source_folder=args.source,
        output_folder=args.output,
        train_ratio=args.train_ratio,
        copy_files=not args.move
    )
    
    print(f"\n✅ Organization Complete!")
    print(f"  Training images: {org_stats['train']:,}")
    print(f"  Gallery images: {org_stats['gallery']:,}")
    print(f"  Query images: {org_stats['query']:,}")
    print(f"  Distractors (in gallery): {org_stats['distractors']:,}")
    print(f"  Junk (separate folder): {org_stats['junk']:,}")
    
    print(f"\n📁 Created structure at: {args.output}")
    print("  ├── bounding_box_train/")
    print("  ├── bounding_box_test/  (gallery)")
    print("  ├── query/")
    print("  └── junk_images/")


if __name__ == "__main__":
    main()
