"""
Quick Win #1: Add confidence scores and validation timestamps to identity files.
This improves data quality and enables belief revision.
"""
import json
from pathlib import Path
from datetime import datetime
import shutil

def add_confidence_to_identity_files():
    """
    Add confidence scores and validation timestamps to all identity files.
    Creates backups before modifying.
    """
    persona_dir = Path("data/persona")
    
    files_to_update = [
        'self_model.json',
        'self_concept.json',
        'narrative_identity.json'
    ]
    
    for filename in files_to_update:
        filepath = persona_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  {filename} not found, skipping")
            continue
        
        print(f"\n📁 Processing {filename}...")
        
        # Load current data
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Backup original
        backup_path = filepath.with_suffix('.json.backup')
        shutil.copy(filepath, backup_path)
        print(f"   📦 Backed up to {backup_path.name}")
        
        # Add metadata
        updated_data = add_metadata_recursive(data, filename)
        
        # Save updated version
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Updated with confidence scores and timestamps")
        print(f"   📊 Added metadata to {count_metadata_fields(updated_data)} fields")

def add_metadata_recursive(obj, filename='', default_confidence=0.5):
    """
    Recursively add confidence and validation timestamps.
    """
    if isinstance(obj, dict):
        # Add version info at root level
        if not obj.get('version'):
            obj['version'] = '2.0.0'
            obj['last_updated'] = datetime.now().isoformat()
        
        # Process all dict values
        for key, value in list(obj.items()):
            if key in ['version', 'last_updated', 'timestamp', 'created_at']:
                continue
                
            if isinstance(value, dict):
                # For nested objects that represent beliefs/traits/concepts
                if key not in ['metadata', 'stats', 'config']:
                    # Add confidence if not present
                    if 'confidence' not in value:
                        value['confidence'] = default_confidence
                    if 'last_validated' not in value:
                        value['last_validated'] = datetime.now().isoformat()
                
                # Recurse
                obj[key] = add_metadata_recursive(value, filename, default_confidence)
            
            elif isinstance(value, list):
                # Process list items
                obj[key] = [add_metadata_recursive(item, filename, default_confidence) 
                           for item in value]
        
        return obj
    
    elif isinstance(obj, list):
        return [add_metadata_recursive(item, filename, default_confidence) for item in obj]
    
    else:
        return obj

def count_metadata_fields(obj):
    """Count how many fields have metadata."""
    count = 0
    
    if isinstance(obj, dict):
        if 'confidence' in obj:
            count += 1
        for value in obj.values():
            count += count_metadata_fields(value)
    elif isinstance(obj, list):
        for item in obj:
            count += count_metadata_fields(item)
    
    return count

if __name__ == "__main__":
    print("="*60)
    print("QUICK WIN #1: Adding Confidence Scores to Identity Files")
    print("="*60)
    add_confidence_to_identity_files()
    print("\n" + "="*60)
    print("✅ COMPLETE - Identity files upgraded with metadata")
    print("="*60)
