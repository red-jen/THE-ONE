"""
Market-1501 Dataset Handler
============================
Complete utilities for loading, parsing, and preparing Market-1501 for Re-ID training.

Dataset Structure:
- bounding_box_train: 12,936 images, 751 identities (for training)
- bounding_box_test: 19,732 images, 750 identities (gallery)
- query: 3,368 images (probe images)

Image Naming: {id}_{cam}s{seq}_{frame}_{detection}.jpg
Example: 0001_c1s1_001051_00.jpg
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import defaultdict
import random

# Optional imports (install if needed)
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not installed. Image loading disabled. Run: pip install Pillow")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Dataset class disabled. Run: pip install torch torchvision")


class ImageInfo(NamedTuple):
    """Parsed information from Market-1501 image filename."""
    path: str
    person_id: int
    camera_id: int
    sequence_id: int
    frame_id: int
    detection_idx: int
    is_distractor: bool  # ID == 0000
    is_junk: bool        # ID == -1


def parse_filename(filepath: str) -> Optional[ImageInfo]:
    """
    Parse Market-1501 filename to extract metadata.
    
    Args:
        filepath: Full path or filename like '0001_c1s1_001051_00.jpg'
    
    Returns:
        ImageInfo namedtuple or None if parsing fails
    """
    filename = os.path.basename(filepath)
    
    # Pattern: {id}_c{cam}s{seq}_{frame}_{det}.jpg
    pattern = r'(-?\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.jpg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    person_id = int(match.group(1))
    camera_id = int(match.group(2))
    sequence_id = int(match.group(3))
    frame_id = int(match.group(4))
    detection_idx = int(match.group(5))
    
    return ImageInfo(
        path=filepath,
        person_id=person_id,
        camera_id=camera_id,
        sequence_id=sequence_id,
        frame_id=frame_id,
        detection_idx=detection_idx,
        is_distractor=(person_id == 0),
        is_junk=(person_id == -1)
    )


def load_dataset_split(folder_path: str, 
                       include_distractors: bool = False,
                       include_junk: bool = False) -> List[ImageInfo]:
    """
    Load all images from a Market-1501 split folder.
    
    Args:
        folder_path: Path to bounding_box_train, bounding_box_test, or query
        include_distractors: Include ID=0000 images (default False)
        include_junk: Include ID=-1 images (default False)
    
    Returns:
        List of ImageInfo objects
    """
    images = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    for img_file in folder.glob("*.jpg"):
        info = parse_filename(str(img_file))
        if info is None:
            continue
        
        # Filter based on flags
        if info.is_distractor and not include_distractors:
            continue
        if info.is_junk and not include_junk:
            continue
        
        images.append(info)
    
    return images


def get_dataset_statistics(images: List[ImageInfo]) -> Dict:
    """
    Compute statistics for a list of images.
    
    Returns:
        Dictionary with dataset statistics
    """
    if not images:
        return {"error": "No images provided"}
    
    person_ids = set()
    camera_ids = set()
    images_per_person = defaultdict(int)
    images_per_camera = defaultdict(int)
    cameras_per_person = defaultdict(set)
    
    for img in images:
        if not img.is_distractor and not img.is_junk:
            person_ids.add(img.person_id)
            camera_ids.add(img.camera_id)
            images_per_person[img.person_id] += 1
            images_per_camera[img.camera_id] += 1
            cameras_per_person[img.person_id].add(img.camera_id)
    
    return {
        "total_images": len(images),
        "num_identities": len(person_ids),
        "num_cameras": len(camera_ids),
        "camera_ids": sorted(camera_ids),
        "avg_images_per_person": sum(images_per_person.values()) / len(person_ids) if person_ids else 0,
        "min_images_per_person": min(images_per_person.values()) if images_per_person else 0,
        "max_images_per_person": max(images_per_person.values()) if images_per_person else 0,
        "avg_cameras_per_person": sum(len(c) for c in cameras_per_person.values()) / len(person_ids) if person_ids else 0,
    }


def create_identity_mapping(images: List[ImageInfo]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create continuous label mapping for person IDs.
    
    Market-1501 has non-continuous IDs. This creates a 0-indexed mapping.
    
    Returns:
        (id_to_label, label_to_id) dictionaries
    """
    unique_ids = sorted(set(img.person_id for img in images 
                           if not img.is_distractor and not img.is_junk))
    
    id_to_label = {pid: idx for idx, pid in enumerate(unique_ids)}
    label_to_id = {idx: pid for pid, idx in id_to_label.items()}
    
    return id_to_label, label_to_id


def group_by_identity(images: List[ImageInfo]) -> Dict[int, List[ImageInfo]]:
    """Group images by person identity."""
    groups = defaultdict(list)
    for img in images:
        if not img.is_distractor and not img.is_junk:
            groups[img.person_id].append(img)
    return dict(groups)


def group_by_camera(images: List[ImageInfo]) -> Dict[int, List[ImageInfo]]:
    """Group images by camera ID."""
    groups = defaultdict(list)
    for img in images:
        groups[img.camera_id].append(img)
    return dict(groups)


def split_query_gallery(test_images: List[ImageInfo], 
                        query_images: List[ImageInfo]) -> Tuple[List[ImageInfo], List[ImageInfo]]:
    """
    Prepare query and gallery sets for evaluation.
    
    For Re-ID evaluation:
    - Query: probe images to search for
    - Gallery: database of images to search in (excludes same camera shots of query)
    
    Returns:
        (query_set, gallery_set)
    """
    # Gallery is all test images except those that would be "junk" for all queries
    gallery = [img for img in test_images if not img.is_junk]
    query = [img for img in query_images if not img.is_junk]
    
    return query, gallery


# =============================================================================
# PyTorch Dataset Class (if torch is available)
# =============================================================================

if HAS_TORCH and HAS_PIL:
    
    class Market1501Dataset(Dataset):
        """
        PyTorch Dataset for Market-1501.
        
        Usage:
            dataset = Market1501Dataset(
                root='/path/to/Market-1501-v15.09.15',
                split='train',  # 'train', 'query', 'gallery'
                transform=transforms.Compose([...])
            )
        """
        
        def __init__(self, 
                     root: str,
                     split: str = 'train',
                     transform=None,
                     target_size: Tuple[int, int] = (256, 128)):  # H x W
            """
            Args:
                root: Root directory of Market-1501 dataset
                split: 'train', 'query', or 'gallery'
                transform: torchvision transforms to apply
                target_size: (height, width) to resize images
            """
            self.root = Path(root)
            self.split = split
            self.target_size = target_size
            
            # Default transforms if none provided
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transform
            
            # Load appropriate split
            if split == 'train':
                folder = self.root / 'bounding_box_train'
            elif split == 'gallery':
                folder = self.root / 'bounding_box_test'
            elif split == 'query':
                folder = self.root / 'query'
            else:
                raise ValueError(f"Invalid split: {split}. Use 'train', 'query', or 'gallery'")
            
            self.images = load_dataset_split(str(folder))
            self.id_to_label, self.label_to_id = create_identity_mapping(self.images)
            
            print(f"Loaded {split} split: {len(self.images)} images, "
                  f"{len(self.id_to_label)} identities")
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_info = self.images[idx]
            
            # Load image
            image = Image.open(img_info.path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label = self.id_to_label.get(img_info.person_id, -1)
            
            return {
                'image': image,
                'label': label,
                'person_id': img_info.person_id,
                'camera_id': img_info.camera_id,
                'path': img_info.path
            }
    
    
    class TripletMarket1501(Dataset):
        """
        Triplet sampling dataset for Re-ID training.
        
        Returns (anchor, positive, negative) triplets where:
        - Anchor & Positive: same identity, different cameras
        - Negative: different identity
        """
        
        def __init__(self, 
                     root: str,
                     transform=None,
                     target_size: Tuple[int, int] = (256, 128)):
            
            self.root = Path(root)
            self.target_size = target_size
            
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transform
            
            # Load training data
            folder = self.root / 'bounding_box_train'
            self.images = load_dataset_split(str(folder))
            
            # Group by identity for efficient triplet mining
            self.by_identity = group_by_identity(self.images)
            self.identities = list(self.by_identity.keys())
            
            # Also group by identity AND camera
            self.by_id_camera = defaultdict(lambda: defaultdict(list))
            for img in self.images:
                if not img.is_distractor and not img.is_junk:
                    self.by_id_camera[img.person_id][img.camera_id].append(img)
            
            print(f"Triplet Dataset: {len(self.images)} images, {len(self.identities)} identities")
        
        def __len__(self):
            return len(self.images)
        
        def _load_image(self, img_info: ImageInfo):
            image = Image.open(img_info.path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        
        def __getitem__(self, idx):
            # Anchor
            anchor_info = self.images[idx]
            anchor_id = anchor_info.person_id
            anchor_cam = anchor_info.camera_id
            
            # Positive: same identity, preferably different camera
            positive_candidates = [
                img for img in self.by_identity[anchor_id]
                if img.camera_id != anchor_cam or img.path != anchor_info.path
            ]
            if not positive_candidates:
                positive_candidates = self.by_identity[anchor_id]
            positive_info = random.choice(positive_candidates)
            
            # Negative: different identity
            neg_id = random.choice([i for i in self.identities if i != anchor_id])
            negative_info = random.choice(self.by_identity[neg_id])
            
            return {
                'anchor': self._load_image(anchor_info),
                'positive': self._load_image(positive_info),
                'negative': self._load_image(negative_info),
                'anchor_id': anchor_id,
                'positive_id': positive_info.person_id,
                'negative_id': neg_id
            }


# =============================================================================
# Evaluation Utilities
# =============================================================================

def compute_distance_matrix(query_features: 'np.ndarray', 
                           gallery_features: 'np.ndarray',
                           metric: str = 'euclidean') -> 'np.ndarray':
    """
    Compute pairwise distance matrix between query and gallery.
    
    Args:
        query_features: (num_query, feature_dim) array
        gallery_features: (num_gallery, feature_dim) array
        metric: 'euclidean' or 'cosine'
    
    Returns:
        (num_query, num_gallery) distance matrix
    """
    if not HAS_PIL:
        raise ImportError("NumPy required. pip install numpy")
    
    import numpy as np
    
    if metric == 'euclidean':
        # Efficient euclidean distance computation
        q_sq = np.sum(query_features ** 2, axis=1, keepdims=True)
        g_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True)
        dist = q_sq + g_sq.T - 2 * np.dot(query_features, gallery_features.T)
        dist = np.sqrt(np.maximum(dist, 0))
    
    elif metric == 'cosine':
        # Normalize then compute cosine distance
        q_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        g_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
        similarity = np.dot(q_norm, g_norm.T)
        dist = 1 - similarity
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return dist


def evaluate_reid(query_features: 'np.ndarray',
                  gallery_features: 'np.ndarray',
                  query_ids: List[int],
                  gallery_ids: List[int],
                  query_cams: List[int],
                  gallery_cams: List[int],
                  metric: str = 'euclidean') -> Dict:
    """
    Evaluate Re-ID performance with CMC and mAP metrics.
    
    IMPORTANT: Following Market-1501 protocol:
    - Exclude gallery images with same ID AND same camera as query
    - These are "junk" for that specific query
    
    Returns:
        Dictionary with Rank-1, Rank-5, Rank-10, mAP scores
    """
    import numpy as np
    
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)
    
    num_query = len(query_ids)
    all_cmc = []
    all_ap = []
    
    for q_idx in range(num_query):
        q_id = query_ids[q_idx]
        q_cam = query_cams[q_idx]
        
        # Sort gallery by distance
        order = np.argsort(dist_matrix[q_idx])
        
        # Remove junk (same camera, same ID)
        valid_mask = ~((np.array(gallery_ids) == q_id) & (np.array(gallery_cams) == q_cam))
        
        # Get matches (same ID, different camera)
        match_mask = (np.array(gallery_ids) == q_id) & valid_mask
        
        # Apply mask to sorted indices
        valid_indices = order[valid_mask[order]]
        match_flags = match_mask[valid_indices]
        
        if not np.any(match_flags):
            continue  # No valid matches for this query
        
        # CMC curve
        cmc = np.cumsum(match_flags) > 0
        all_cmc.append(cmc)
        
        # Average Precision
        num_matches = np.sum(match_flags)
        positions = np.where(match_flags)[0] + 1  # 1-indexed positions
        precision_at_k = np.arange(1, num_matches + 1) / positions
        ap = np.mean(precision_at_k)
        all_ap.append(ap)
    
    all_cmc = np.array(all_cmc)
    cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_ap)
    
    return {
        'mAP': mAP * 100,
        'Rank-1': cmc[0] * 100 if len(cmc) > 0 else 0,
        'Rank-5': cmc[4] * 100 if len(cmc) > 4 else 0,
        'Rank-10': cmc[9] * 100 if len(cmc) > 9 else 0,
        'num_valid_queries': len(all_ap)
    }


# =============================================================================
# Main - Demo usage
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market-1501 Dataset Utilities")
    parser.add_argument('--root', type=str, required=True,
                        help='Path to Market-1501-v15.09.15 folder')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Market-1501 Dataset Analysis")
    print("=" * 60)
    
    root = Path(args.root)
    
    # Analyze each split
    splits = {
        'Training': root / 'bounding_box_train',
        'Gallery (Test)': root / 'bounding_box_test', 
        'Query': root / 'query'
    }
    
    for name, folder in splits.items():
        if folder.exists():
            images = load_dataset_split(str(folder))
            stats = get_dataset_statistics(images)
            
            print(f"\n{name}:")
            print(f"  Total images: {stats['total_images']:,}")
            print(f"  Identities: {stats['num_identities']}")
            print(f"  Cameras: {stats['camera_ids']}")
            print(f"  Avg images/person: {stats['avg_images_per_person']:.1f}")
            print(f"  Min/Max per person: {stats['min_images_per_person']}/{stats['max_images_per_person']}")
        else:
            print(f"\n{name}: Folder not found at {folder}")
    
    print("\n" + "=" * 60)
