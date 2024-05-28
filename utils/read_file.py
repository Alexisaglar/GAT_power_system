import h5py

def print_h5_structure(file):
    with h5py.File(file, 'r') as f:
        def print_group(name, obj):
            print(name)
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset shape: {obj.shape}")
            else:
                print(f"  Group with {len(obj.keys())} keys")

        f.visititems(print_group)

# Example usage:
print_h5_structure('raw_data/network_results.h5')
