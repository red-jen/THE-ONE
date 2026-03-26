"""
Market-1501 Quick Start Guide
==============================
Example scripts to get started with the dataset.
"""

from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.market1501_dataset import (
    load_dataset_split,
    get_dataset_statistics,
    parse_filename,
    group_by_identity,
    create_identity_mapping
)


def example_basic_usage():
    """Basic usage: Load and analyze the dataset."""
    
    # TODO: Update this path to your Market-1501 location
    DATASET_ROOT = r"C:\path\to\Market-1501-v15.09.15"
    
    # Example 1: Parse a single filename
    print("=" * 50)
    print("Example 1: Parse filename")
    print("=" * 50)
    
    sample_name = "0001_c1s1_001051_00.jpg"
    info = parse_filename(sample_name)
    print(f"Filename: {sample_name}")
    print(f"  Person ID: {info.person_id}")
    print(f"  Camera: {info.camera_id}")
    print(f"  Sequence: {info.sequence_id}")
    print(f"  Frame: {info.frame_id}")
    print(f"  Detection index: {info.detection_idx}")
    
    
def example_load_splits():
    """Load and analyze different splits."""
    
    # TODO: Update this path
    DATASET_ROOT = Path(r"C:\path\to\Market-1501-v15.09.15")
    
    print("\n" + "=" * 50)
    print("Example 2: Load dataset splits")
    print("=" * 50)
    
    # Load training data
    train_folder = DATASET_ROOT / "bounding_box_train"
    if train_folder.exists():
        train_images = load_dataset_split(str(train_folder))
        stats = get_dataset_statistics(train_images)
        
        print(f"\nTraining Set:")
        print(f"  Images: {stats['total_images']:,}")
        print(f"  Identities: {stats['num_identities']}")
        print(f"  Cameras: {stats['num_cameras']}")
    else:
        print(f"Training folder not found at: {train_folder}")


def example_identity_analysis():
    """Analyze identities in the dataset."""
    
    # TODO: Update this path  
    DATASET_ROOT = Path(r"C:\path\to\Market-1501-v15.09.15")
    
    print("\n" + "=" * 50)
    print("Example 3: Identity Analysis")
    print("=" * 50)
    
    train_folder = DATASET_ROOT / "bounding_box_train"
    if not train_folder.exists():
        print("Update DATASET_ROOT path first!")
        return
    
    train_images = load_dataset_split(str(train_folder))
    
    # Group by identity
    by_identity = group_by_identity(train_images)
    
    # Find identities with most/least images
    id_counts = {pid: len(imgs) for pid, imgs in by_identity.items()}
    
    # Sort by count
    sorted_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 identities (most images):")
    for pid, count in sorted_ids[:5]:
        # Show which cameras captured this person
        cameras = set(img.camera_id for img in by_identity[pid])
        print(f"  ID {pid:04d}: {count} images, cameras: {sorted(cameras)}")
    
    print("\nBottom 5 identities (least images):")
    for pid, count in sorted_ids[-5:]:
        cameras = set(img.camera_id for img in by_identity[pid])
        print(f"  ID {pid:04d}: {count} images, cameras: {sorted(cameras)}")


def example_pytorch_dataloader():
    """Example using PyTorch DataLoader."""
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError:
        print("PyTorch not installed. Run: pip install torch torchvision")
        return
    
    from data.market1501_dataset import Market1501Dataset, TripletMarket1501
    
    # TODO: Update this path
    DATASET_ROOT = r"C:\path\to\Market-1501-v15.09.15"
    
    print("\n" + "=" * 50)
    print("Example 4: PyTorch DataLoader")
    print("=" * 50)
    
    # Custom transforms for training
    train_transforms = transforms.Compose([
        transforms.Resize((256, 128)),  # Standard Re-ID size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.4))  # Cutout augmentation
    ])
    
    # Create dataset
    train_dataset = Market1501Dataset(
        root=DATASET_ROOT,
        split='train',
        transform=train_transforms
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Iterate one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Images shape: {batch['image'].shape}")  # [32, 3, 256, 128]
    print(f"  Labels: {batch['label'][:5]}...")  # First 5 labels
    print(f"  Person IDs: {batch['person_id'][:5]}...")


def example_triplet_training():
    """Example triplet training setup."""
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed.")
        return
    
    from data.market1501_dataset import TripletMarket1501
    
    # TODO: Update this path
    DATASET_ROOT = r"C:\path\to\Market-1501-v15.09.15"
    
    print("\n" + "=" * 50)
    print("Example 5: Triplet Training Setup")
    print("=" * 50)
    
    # Create triplet dataset
    triplet_dataset = TripletMarket1501(root=DATASET_ROOT)
    
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Get one triplet batch
    batch = next(iter(triplet_loader))
    print(f"\nTriplet batch:")
    print(f"  Anchor shape: {batch['anchor'].shape}")
    print(f"  Positive shape: {batch['positive'].shape}")
    print(f"  Negative shape: {batch['negative'].shape}")
    print(f"  Anchor IDs: {batch['anchor_id'][:5]}")
    print(f"  Negative IDs: {batch['negative_id'][:5]}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    
    # Uncomment these after updating DATASET_ROOT:
    # example_load_splits()
    # example_identity_analysis()
    # example_pytorch_dataloader()
    # example_triplet_training()
