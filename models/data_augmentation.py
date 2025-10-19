"""
Data Augmentation Module for Small Dataset Enhancement

This module provides data augmentation techniques to improve model generalization
when training datasets are small (< 300 samples). It implements Gaussian noise
addition and SMOTE for class balancing.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    """Results from data augmentation process"""
    original_size: int
    augmented_size: int
    augmentation_methods: list
    class_distribution_before: Dict[int, int]
    class_distribution_after: Dict[int, int]


class DataAugmenter:
    """
    Data augmentation class for enhancing small datasets.
    
    Applies Gaussian noise and SMOTE techniques to create synthetic samples
    and balance class distributions, helping models generalize better.
    """
    
    def __init__(self, noise_std: float = 0.15, smote_k_neighbors: int = 5):
        """
        Initialize DataAugmenter with configuration parameters.
        
        Args:
            noise_std: Standard deviation for Gaussian noise (default: 0.15)
            smote_k_neighbors: Number of neighbors for SMOTE algorithm (default: 5)
        """
        self.noise_std = noise_std
        self.smote_k_neighbors = smote_k_neighbors
        self.augmentation_result: Optional[AugmentationResult] = None
        
    def should_augment(self, n_samples: int, threshold: int = 300) -> bool:
        """
        Determine if dataset needs augmentation based on sample count.
        
        Args:
            n_samples: Number of samples in the dataset
            threshold: Minimum sample count threshold (default: 300)
            
        Returns:
            True if augmentation is needed (n_samples < threshold), False otherwise
        """
        return n_samples < threshold
    
    def add_gaussian_noise(self, X: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to features to create variations.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            std: Standard deviation for noise (uses self.noise_std if None)
            
        Returns:
            Feature matrix with added Gaussian noise
        """
        if std is None:
            std = self.noise_std
            
        noise = np.random.normal(0, std, X.shape)
        X_noisy = X + noise
        
        logger.info(f"Added Gaussian noise with std={std} to {X.shape[0]} samples")
        return X_noisy
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) for class balancing.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Returns:
            Tuple of (X_resampled, y_resampled) with balanced classes
            
        Note:
            Falls back to returning original data if SMOTE fails
        """
        try:
            # Check if we have enough samples for SMOTE
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            
            # Adjust k_neighbors if needed
            k_neighbors = min(self.smote_k_neighbors, min_class_count - 1)
            
            if k_neighbors < 1:
                logger.warning(
                    f"Insufficient samples for SMOTE (min class count: {min_class_count}). "
                    "Skipping SMOTE augmentation."
                )
                return X, y
            
            # Apply SMOTE
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(
                f"Applied SMOTE: {X.shape[0]} samples -> {X_resampled.shape[0]} samples"
            )
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed with error: {str(e)}. Returning original data.")
            return X, y
    
    def time_shift(self, X: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """
        Apply time-shift augmentation by shifting features by ±max_shift positions.
        
        This creates variations by simulating temporal shifts in the data,
        useful for time-series or sequential features.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            max_shift: Maximum number of positions to shift (default: 2)
            
        Returns:
            Feature matrix with time-shifted values
        """
        X_shifted = X.copy()
        
        # Apply random shift to each sample
        for i in range(len(X)):
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                X_shifted[i] = np.roll(X[i], shift)
        
        logger.info(f"Applied time-shift augmentation (±{max_shift} timesteps) to {X.shape[0]} samples")
        return X_shifted
    
    def feature_dropout(self, X: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """
        Apply feature dropout by randomly zeroing out features.
        
        This helps models learn robust representations that don't rely on
        specific features always being present.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            dropout_rate: Probability of dropping each feature (default: 0.1)
            
        Returns:
            Feature matrix with randomly dropped features
        """
        X_dropout = X.copy()
        
        # Create dropout mask (1 = keep, 0 = drop)
        mask = np.random.binomial(1, 1 - dropout_rate, X.shape)
        X_dropout *= mask
        
        n_dropped = np.sum(mask == 0)
        logger.info(f"Applied feature dropout ({dropout_rate:.0%} rate) - dropped {n_dropped} feature values")
        return X_dropout
    
    def _validate_distribution(self, y_original: np.ndarray, y_augmented: np.ndarray, 
                              tolerance: float = 0.05) -> bool:
        """
        Validate that augmented data preserves label distribution within tolerance.
        
        Args:
            y_original: Original target labels
            y_augmented: Augmented target labels
            tolerance: Maximum allowed difference in class proportions (default: 0.05)
            
        Returns:
            True if distribution is preserved within tolerance, False otherwise
        """
        # Calculate class distributions
        unique_classes = np.unique(np.concatenate([y_original, y_augmented]))
        
        orig_dist = np.zeros(len(unique_classes))
        aug_dist = np.zeros(len(unique_classes))
        
        for idx, cls in enumerate(unique_classes):
            orig_dist[idx] = np.sum(y_original == cls) / len(y_original)
            aug_dist[idx] = np.sum(y_augmented == cls) / len(y_augmented)
        
        # Calculate maximum difference
        max_diff = np.max(np.abs(orig_dist - aug_dist))
        
        if max_diff > tolerance:
            logger.warning(
                f"Label distribution shifted by {max_diff:.2%} (tolerance: {tolerance:.2%})"
            )
            for idx, cls in enumerate(unique_classes):
                logger.warning(
                    f"  Class {cls}: {orig_dist[idx]:.2%} -> {aug_dist[idx]:.2%} "
                    f"(diff: {abs(orig_dist[idx] - aug_dist[idx]):.2%})"
                )
            return False
        
        logger.info(f"Label distribution preserved within {tolerance:.2%} tolerance (max diff: {max_diff:.2%})")
        return True
    
    def _validate_ranges(self, X_augmented: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """
        Validate that augmented data maintains realistic value ranges.
        
        Clips augmented values to the min/max range of original data.
        
        Args:
            X_augmented: Augmented feature matrix
            X_original: Original feature matrix
            
        Returns:
            Validated and clipped feature matrix
        """
        # Calculate min and max for each feature
        feature_mins = np.min(X_original, axis=0)
        feature_maxs = np.max(X_original, axis=0)
        
        # Clip augmented data to original ranges
        X_validated = np.clip(X_augmented, feature_mins, feature_maxs)
        
        # Check how many values were clipped
        n_clipped = np.sum(X_augmented != X_validated)
        if n_clipped > 0:
            logger.info(f"Clipped {n_clipped} values to maintain realistic ranges")
        
        return X_validated
    
    def augment(self, X: np.ndarray, y: np.ndarray, 
                threshold: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive data augmentation with multiple techniques.
        
        This method implements adaptive augmentation based on dataset size:
        - <200 samples: 3x augmentation
        - 200-300 samples: 2x augmentation
        - >=300 samples: no augmentation
        
        Augmentation techniques applied:
        1. Gaussian noise (std=0.15)
        2. Time-shift augmentation (±2 timesteps)
        3. Feature dropout (10% rate)
        4. SMOTE for class balancing
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            threshold: Sample count threshold for augmentation (default: 300)
            
        Returns:
            Tuple of (X_augmented, y_augmented) with enhanced dataset
        """
        original_size = X.shape[0]
        methods_used = []
        
        # Get original class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_dist_before = dict(zip(unique_classes.tolist(), class_counts.tolist()))
        
        # Check if augmentation is needed
        if not self.should_augment(original_size, threshold):
            logger.info(
                f"Dataset has {original_size} samples (>= {threshold}). "
                "Skipping augmentation."
            )
            self.augmentation_result = AugmentationResult(
                original_size=original_size,
                augmented_size=original_size,
                augmentation_methods=[],
                class_distribution_before=class_dist_before,
                class_distribution_after=class_dist_before
            )
            return X, y
        
        # Determine augmentation factor based on dataset size
        if original_size < 200:
            augmentation_factor = 3
            logger.info(f"Dataset has {original_size} samples (<200). Target: 3x augmentation")
        else:  # 200-300 samples
            augmentation_factor = 2
            logger.info(f"Dataset has {original_size} samples (200-300). Target: 2x augmentation")
        
        target_size = original_size * augmentation_factor
        
        # Store original data for range validation
        X_original = X.copy()
        y_original = y.copy()
        
        # Initialize with original data
        X_aug_list = [X.copy()]
        y_aug_list = [y.copy()]
        
        # Apply multiple augmentation techniques to reach target size
        remaining_samples = target_size - original_size
        
        # Technique 1: Gaussian noise (increased to 0.15)
        if remaining_samples > 0:
            X_noise = self.add_gaussian_noise(X)
            X_aug_list.append(X_noise)
            y_aug_list.append(y.copy())
            methods_used.append('gaussian_noise')
            remaining_samples -= original_size
        
        # Technique 2: Time-shift augmentation
        if remaining_samples > 0:
            X_shifted = self.time_shift(X, max_shift=2)
            X_aug_list.append(X_shifted)
            y_aug_list.append(y.copy())
            methods_used.append('time_shift')
            remaining_samples -= original_size
        
        # Technique 3: Feature dropout
        if remaining_samples > 0:
            X_dropout = self.feature_dropout(X, dropout_rate=0.1)
            X_aug_list.append(X_dropout)
            y_aug_list.append(y.copy())
            methods_used.append('feature_dropout')
            remaining_samples -= original_size
        
        # Combine all augmented data
        X_combined = np.vstack(X_aug_list)
        y_combined = np.hstack(y_aug_list)
        
        # Trim to target size if we exceeded it
        if len(X_combined) > target_size:
            indices = np.random.choice(len(X_combined), target_size, replace=False)
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            logger.info(f"Trimmed augmented data to target size: {target_size}")
        
        # Apply SMOTE for class balancing
        X_augmented, y_augmented = self.apply_smote(X_combined, y_combined)
        if X_augmented.shape[0] > X_combined.shape[0]:
            methods_used.append('smote')
        
        # Validate ranges
        X_augmented = self._validate_ranges(X_augmented, X_original)
        
        # Validate label distribution (within 5% tolerance)
        self._validate_distribution(y_original, y_augmented, tolerance=0.05)
        
        # Get final class distribution
        unique_classes_after, class_counts_after = np.unique(y_augmented, return_counts=True)
        class_dist_after = dict(zip(unique_classes_after.tolist(), class_counts_after.tolist()))
        
        # Store augmentation results
        augmented_size = X_augmented.shape[0]
        self.augmentation_result = AugmentationResult(
            original_size=original_size,
            augmented_size=augmented_size,
            augmentation_methods=methods_used,
            class_distribution_before=class_dist_before,
            class_distribution_after=class_dist_after
        )
        
        # Report results
        self._report_augmentation()
        
        return X_augmented, y_augmented
    
    def _report_augmentation(self):
        """Report augmentation statistics to logger."""
        if self.augmentation_result is None:
            return
        
        result = self.augmentation_result
        
        logger.info("=" * 60)
        logger.info("DATA AUGMENTATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Original dataset size: {result.original_size}")
        logger.info(f"Augmented dataset size: {result.augmented_size}")
        logger.info(f"Augmentation ratio: {result.augmented_size / result.original_size:.2f}x")
        logger.info(f"Methods applied: {', '.join(result.augmentation_methods)}")
        logger.info("")
        logger.info("Class distribution BEFORE augmentation:")
        for class_label, count in result.class_distribution_before.items():
            logger.info(f"  Class {class_label}: {count} samples")
        logger.info("")
        logger.info("Class distribution AFTER augmentation:")
        for class_label, count in result.class_distribution_after.items():
            logger.info(f"  Class {class_label}: {count} samples")
        logger.info("=" * 60)
    
    def get_augmentation_result(self) -> Optional[AugmentationResult]:
        """
        Get the results from the last augmentation operation.
        
        Returns:
            AugmentationResult object or None if no augmentation has been performed
        """
        return self.augmentation_result
