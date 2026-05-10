import pandas as pd
from pathlib import Path

def check_database_status():
    # Path to your master database
    db_path = Path("data/small_world/axelrod_sw_master_results.parquet")
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    # Load data
    df = pd.read_parquet(db_path)
    
    total_entries = len(df)
    non_frozen_df = df[df['is_frozen'] == False]
    frozen_df = df[df['is_frozen'] == True]
    
    num_non_frozen = len(non_frozen_df)
    num_frozen = len(frozen_df)

    unique_ps = sorted(df['p'].unique().tolist())
    
    # Calculate percentage
    percentage = (num_non_frozen / total_entries) * 100 if total_entries > 0 else 0

    print("="*65)
    print("AXELROD DATABASE STATUS REPORT")
    print("="*65)
    print(f"Total realizations collected: {total_entries}")
    print(f"Frozen realizations:         {num_frozen}")
    print(f"NOT FROZEN realizations:     {num_non_frozen} ({percentage:.2f}%)")
    print(f"unique p (rewireing prob ) values: {len(unique_ps)}")
    print(unique_ps)
    print("-" * 65)

    # --- Section 1: Non-Frozen Breakdown ---
    if num_non_frozen > 0:
        print("BREAKDOWN OF NON-FROZEN RUNS (Incomplete):")
        breakdown_nf = non_frozen_df.groupby(['F', 'k', 'q']).size().reset_index(name='count')
        print(breakdown_nf.sort_values(['F', 'k', 'q']).to_string(index=False))
        print("-" * 65)
    else:
        print("All entries in the database are frozen. Great!")
        print("-" * 65)

    # --- Section 2: Convergence Speed (For Frozen Runs) ---
    if num_frozen > 0:
        print("CONVERGENCE SPEED (Average steps for frozen runs):")
        # Group by parameters and calculate mean and max steps
        speed_stats = frozen_df.groupby(['F', 'k', 'q'])['steps_to_freeze'].agg(['mean', 'max']).reset_index()
        
        # Rename columns for clarity
        speed_stats.columns = ['F', 'k', 'q', 'Avg Steps', 'Max Steps']
        
        # Format numbers for better readability (thousands separator)
        pd.options.display.float_format = '{:,.0f}'.format
        
        # Sort and print
        print(speed_stats.sort_values(['F', 'k', 'q']).to_string(index=False))
        
        print("-" * 65)
        print("Note: Convergence steps usually peak near the phase transition.")
    else:
        print("No frozen data available to calculate convergence speed.")
    
    print("="*65)

if __name__ == "__main__":
    check_database_status()