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
    
    def __init__(self, noise_std: float = 0.01, smote_k_neighbors: int = 5):
        """
        Initialize DataAugmenter with configuration parameters.
        
        Args:
            noise_std: Standard deviation for Gaussian noise (default: 0.01)
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
        Apply comprehensive data augmentation combining noise and SMOTE.
        
        This method:
        1. Checks if augmentation is needed
        2. Adds Gaussian noise to create variations
        3. Applies SMOTE for class balancing
        4. Validates augmented data ranges
        5. Reports augmentation statistics
        
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
        
        logger.info(
            f"Dataset has {original_size} samples (< {threshold}). "
            "Applying augmentation..."
        )
        
        # Store original data for range validation
        X_original = X.copy()
        
        # Step 1: Add Gaussian noise
        X_augmented = self.add_gaussian_noise(X)
        methods_used.append('gaussian_noise')
        
        # Step 2: Apply SMOTE for class balancing
        X_augmented, y_augmented = self.apply_smote(X_augmented, y)
        if X_augmented.shape[0] > X.shape[0]:
            methods_used.append('smote')
        
        # Step 3: Validate ranges
        X_augmented = self._validate_ranges(X_augmented, X_original)
        
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
