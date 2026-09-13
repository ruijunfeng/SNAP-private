from sklearn.utils import resample, shuffle


def generate_balance_splits(dataset, train_indices, val_indices, test_indices, label_key="target"):
    """Balance dataset based on label distribution and save balanced indices."""
    
    def balance_data_by_resample(data, indices, label_key="target"):
        """Balances the dataset by undersampling the majority class or oversampling the minority class, returning the balanced indices."""
        # Split data and indices into different classes
        label_to_samples = {}
        label_to_indices = {}
        for sample, index in zip(data, indices):
            label = sample[label_key]
            if label not in label_to_samples:
                label_to_samples[label] = []
                label_to_indices[label] = []
            label_to_samples[label].append(sample)
            label_to_indices[label].append(index)
        
        # Find the minimum number of samples across all classes
        min_samples = min(len(samples) for samples in label_to_samples.values())
        
        # Resample each class to match the minimum number of samples, and track their indices
        balanced_indices = []
        for label, samples in label_to_samples.items():
            if len(samples) > min_samples:
                # Undersample
                balanced_indices_for_label = resample(label_to_indices[label], replace=False, n_samples=min_samples, random_state=42)
            else:
                # No need to resample, use all samples
                balanced_indices_for_label = label_to_indices[label]
            balanced_indices.extend(balanced_indices_for_label)
        
        # Shuffle both data and indices in the same order
        balanced_indices = shuffle(balanced_indices, random_state=42)
        
        return balanced_indices
    
    # Retrieve the data samples corresponding to each set of indices
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]
    
    # Apply balancing to each split
    balanced_train_indices = balance_data_by_resample(train_data, train_indices, label_key)
    balanced_val_indices = balance_data_by_resample(val_data, val_indices, label_key)
    balanced_test_indices = balance_data_by_resample(test_data, test_indices, label_key)
    
    return balanced_train_indices, balanced_val_indices, balanced_test_indices
