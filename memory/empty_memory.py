from pathlib import Path
import shutil

storage_dir = Path("/memory/storage")

if storage_dir.exists():
    shutil.rmtree(storage_dir)
    print("Memory storage deleted successfully.")
else:
    print("Memory storage does not exist.")