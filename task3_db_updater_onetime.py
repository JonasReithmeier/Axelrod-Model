import pandas as pd
from pathlib import Path

def main():
    db_path = Path("data/schelling/schelling_master_results.parquet")
    csv_path = Path("data/schelling/schelling_master_results.csv")
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    # 1. Load the existing database
    df = pd.read_parquet(db_path)
    
    # 2. Add the column if it doesn't exist
    if 'avg_mobility' not in df.columns:
        df['avg_mobility'] = float('nan') # Initialize all with NaN

    # 3. Set avg_mobility to 0.0 ONLY for the runs that are already frozen
    # (The unfrozen ones will stay NaN until your runner finishes them)
    frozen_mask = df['is_frozen'] == True
    df.loc[frozen_mask, 'avg_mobility'] = 0.0

    # 4. Save a backup just in case, then overwrite the master files
    df.to_parquet(db_path.with_suffix('.parquet.bak'))
    df.to_parquet(db_path, index=False)
    df.to_csv(csv_path, index=False)

    frozen_count = frozen_mask.sum()
    print(f"Success! Added 'avg_mobility' column.")
    print(f"Updated {frozen_count} already-frozen runs to avg_mobility = 0.0.")
    print(f"Left {len(df) - frozen_count} unfrozen runs as NaN to be processed by task3_runner.py.")

if __name__ == "__main__":
    main()