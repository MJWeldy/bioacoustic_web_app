#!/usr/bin/env python3
"""
Utility script to populate embedding indices for existing database.
Run this once after updating the database schema to add embedding_index column.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from modules.database import Audio_DB

def populate_embedding_indices(dataset_path: str):
    """
    Populate embedding indices for a dataset.
    
    Args:
        dataset_path: Path to the dataset directory containing database.parquet and embeddings files
    """
    print(f"Processing dataset: {dataset_path}")
    
    # Paths
    database_path = os.path.join(dataset_path, 'database.parquet')
    embeddings_path = os.path.join(dataset_path, 'embeddings.pkl')
    
    # Check if files exist
    if not os.path.exists(database_path):
        print(f"❌ Database not found: {database_path}")
        return False
        
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings not found: {embeddings_path}")
        return False
    
    try:
        # Load database
        print("Loading database...")
        db = Audio_DB()
        db.load_db(database_path)
        print(f"✓ Loaded database with {len(db.df)} clips")
        
        # Load embeddings to get count
        print("Loading embeddings...")
        embeddings = np.load(embeddings_path, allow_pickle=True)
        
        if isinstance(embeddings, np.ndarray):
            embeddings_count = embeddings.shape[0]
        else:
            # Handle pickled list format
            embeddings_count = len(embeddings)
            
        print(f"✓ Found {embeddings_count} embeddings")
        
        # Check current embedding_index status
        null_count = db.df['embedding_index'].null_count()
        print(f"Current null embedding indices: {null_count}")
        
        if null_count > 0:
            # Populate indices
            print("Populating embedding indices...")
            db.populate_embedding_indices_by_order(embeddings_count)
            
            # Save updated database
            print("Saving updated database...")
            db.save_db(database_path)
            print(f"✅ Successfully updated {database_path}")
            
            # Verify
            updated_null_count = db.df['embedding_index'].null_count()
            print(f"Remaining null indices: {updated_null_count}")
            
        else:
            print("✓ All clips already have embedding indices")
            
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python populate_embedding_indices.py /path/to/dataset")
        print("")
        print("This script will:")
        print("1. Load the database.parquet file")
        print("2. Check embeddings.pkl to get count")
        print("3. Populate embedding_index column with sequential indices (0, 1, 2, ...)")
        print("4. Save the updated database")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    success = populate_embedding_indices(dataset_path)
    
    if success:
        print("\n🎉 Embedding indices populated successfully!")
        print("\nYou can now use vector search with:")
        print("  similar_clips, similarities, indices = db.find_similar_clips(embeddings, query_embedding)")
    else:
        print("\n❌ Failed to populate embedding indices")
        sys.exit(1)

if __name__ == "__main__":
    main()