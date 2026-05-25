import pandas as pd
import zlib
import hashlib
from pathlib import Path

def run_diagnostic():
    db_path = Path("data/schelling/schelling_master_results.parquet")
    if not db_path.exists():
        print("Database not found!")
        return

    df = pd.read_parquet(db_path)
    print(f"Total rows in DB: {len(df)}")

    # Get the very first row that is structurally frozen
    frozen_runs = df[df['is_frozen'] == True]
    if frozen_runs.empty:
        print("No completely frozen runs in the database to test.")
        return
        
    row = frozen_runs.iloc[-1] # Grab the most recent one

    print("\n--- DIAGNOSTIC CHECK ---")
    print(f"Target DB Parameters : w={row.width}, q={row.q}, F={row.F}, h={row.h}, T={row.T}")
    print(f"Target DB Seed       : {row.seed}")
    
    # Can we recreate this seed using the current logic?
    found_match = False
    for m in range(1000): # Check up to M=1000
        # This matches the exact string generator in task3_runner
        ctx = f"42_{int(row.F)}_{int(row.q)}_{int(row.width)}_{float(row.h)}_{float(row.T)}_{m}"
        
        legacy_seed = zlib.adler32(ctx.encode())
        new_md5_seed = int(hashlib.md5(ctx.encode()).hexdigest()[:16], 16)
        
        if legacy_seed == row.seed:
            print(f"\n✅ SUCCESS! Adler32 Seed matched perfectly at m={m}")
            found_match = True
            break
        if new_md5_seed == row.seed:
            print(f"\n✅ SUCCESS! MD5 Seed matched perfectly at m={m}")
            found_match = True
            break

    if not found_match:
        print("\n❌ FAILED. The dictionary lookup is breaking because the string generation or Pandas float precision altered the seed.")

if __name__ == "__main__":
    run_diagnostic()