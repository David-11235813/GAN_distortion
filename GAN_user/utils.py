import os

def get_last_subfolder(parent_dir):
    subdirs = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not subdirs: return None
    return os.path.join(parent_dir, sorted(subdirs)[-1])
