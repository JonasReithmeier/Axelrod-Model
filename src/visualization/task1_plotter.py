import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
import yaml
import numpy as np

def plot_axelrod_data():
    # --- 1. Load Data ---
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_path = Path("data/task1/" + config['grid_visualization']['database_name'] + ".parquet")

    if not data_path.exists():
        print(f"Error: Database not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    
    # Filter: Usually, phase transition plots only consider reached equilibrium
    # If you want to include non-frozen ones, comment out the line below.
    df = df[df['is_frozen'] == True]

    if df.empty:
        print("No frozen data available to plot.")
        return

    # --- 2. Setup Output Folder ---
    plot_dir = Path("reportPlots/task1")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Scientific Styling ---
    plt.style.use('seaborn-v0_8-paper') # Professional clean style
    sns.set_context("paper", font_scale=1.5)
    
    # --- 4. Plotting Logic ---
    unique_Fs = sorted(df['F'].unique())
    metrics = [('s_max', 'Largest Cluster Size (Normalized)'), 
               ('s_mean', 'Average Cluster Size (Normalized)')]

    M_min = -1 # realizations (smallest group)

    ploted_widths = [20,30,40,50,100,150]

    #for F in unique_Fs:
    #for F in [3]:
    #    df_f = df[df['F'] == F]
    #    
    #    for metric_key, metric_name in metrics:
    #        plt.figure(figsize=(10, 7))
    #        
    #        # Group by width (N) and q to get statistics
    #        # We use width for the legend label
    #        widths = sorted(df_f['width'].unique())
    #        
    #        #for width in widths:
    #        for width in ploted_widths:
    #        #to recreate graph from paper on these widths
    #            df_w = df_f[df_f['width'] == width]
    #            
    #            # Aggregate data for each q
    #            stats = df_w.groupby('q')[metric_key].agg(['mean', 'std']).reset_index()
    #            number_realisations = len(df_w.groupby('q')[metric_key])
    #            
    #            # Plot line with error bars (standard deviation)
    #            plt.errorbar(
    #                stats['q'], stats['mean'], yerr=stats['std']/math.sqrt(number_realisations),
    #                label=f"N = {width}²",
    #                marker='o', markersize=4, capsize=4, capthick= 1.5, elinewidth=1, linestyle='-'
    #            )###
#
#            # Labels and Aesthetics
    #        plt.title(f"Axelrod Model, regular lattice, M=500 realizations | F = {F} features", pad=20)
    #        plt.xlabel("Number of Traits per Feature (q)")
    #        plt.ylabel(metric_name)
    #        plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
    #        plt.grid(True, which='both', linestyle='--', alpha=0.5)
    #        plt.legend(title="Grid Size", frameon=True)
    #        
    #        # Save Logic
    #        file_name =  "task1_graph_recreation.png"  #f"axelrod_F{F}_{metric_key}.png"
    ##        save_path = plot_dir / file_name
    #        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #        plt.close() # Close figure to free memory
#
    #print(f"Update complete. Plots saved to: {plot_dir}")

    for F in unique_Fs:
        df_f = df[df['F'] == F]
        
        for metric_key, metric_name in metrics:
            plt.figure(figsize=(10, 7))
            
            # Group by width (N) and q to get statistics
            # We use width for the legend label
            widths = sorted(df_f['width'].unique())
            
            #for width in widths:
            for width in ploted_widths:
            #to recreate graph from paper on these widths
                df_w = df_f[df_f['width'] == width]
                
                # Aggregate data for each q
                stats = df_w.groupby('q')[metric_key].agg(['mean', 'std']).reset_index()
                number_realisations = df_w.groupby('q')[metric_key].size().iloc[0]
                print(number_realisations)
                #exit()
                # Plot line with error bars (standard deviation)
                plt.errorbar(
                    stats['q'], stats['mean'], yerr=stats['std']/math.sqrt(number_realisations),
                    label=f"N = {width}²",
                    marker='o', markersize=4, capsize=4, capthick=1.5, elinewidth=1, linestyle='-'
                )

            # Labels and Aesthetics
            plt.title(f"Axelrod Model, regular lattice, {metric_key}, M=500 realizations, F = {F}", pad=20)
            plt.xlabel("Number of Traits per Feature (q)")
            plt.ylabel(metric_name)
            plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(title="Grid Size", frameon=True)
            
            # Save Logic
            file_name =  f"axelrod_F{F}_{metric_key}.png"
            save_path = plot_dir / file_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() # Close figure to free memory

    print(f"Update complete. Plots saved to: {plot_dir}")

    for F in unique_Fs:
        df_f = df[df['F'] == F]
        
        for metric_key, metric_name in metrics:
            plt.figure(figsize=(10, 7))
            
            # Group by width (N) and q to get statistics
            # We use width for the legend label
            widths = sorted(df_f['width'].unique())
            
            #for width in widths:
            for width in ploted_widths:
            #to recreate graph from paper on these widths
                df_w = df_f[df_f['width'] == width]
                
                # Aggregate data for each q
                stats = df_w.groupby('q')[metric_key].agg(['mean', 'std']).reset_index()
                number_realisations = df_w.groupby('q')[metric_key].size().iloc[0]
                print(number_realisations)
                #exit()
                # Plot line with error bars (standard deviation)
                plt.errorbar(
                    stats['q']/math.pow(width,2), stats['mean'], yerr=stats['std']/math.sqrt(number_realisations),
                    label=f"N = {width}²",
                    marker='o', markersize=4, capsize=4, capthick=1.5, elinewidth=1, linestyle='-'
                )

            # Labels and Aesthetics
            plt.title(f"Axelrod Model, regular lattice, {metric_key}, M=500 realizations, F = {F}", pad=20)
            plt.xlabel("Number of Traits per Feature (q)")
            plt.ylabel(metric_name)
            plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(title="Grid Size", frameon=True)
            
            # Save Logic
            file_name =  f"axelrod_F{F}_{metric_key}_normalizedQ.png"
            save_path = plot_dir / file_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() # Close figure to free memory

    print(f"Update complete. Plots saved to: {plot_dir}")

    for F in unique_Fs:
        df_f = df[df['F'] == F]
        
        for metric_key, metric_name in metrics:
            plt.figure(figsize=(10, 7))
            
            # Group by width (N) and q to get statistics
            # We use width for the legend label
            widths = sorted(df_f['width'].unique())
            
            #for width in widths:
            for width in ploted_widths:
            #to recreate graph from paper on these widths
                df_w = df_f[df_f['width'] == width]
                
                # Aggregate data for each q
                stats = df_w.groupby('q')[metric_key].agg(['mean', 'std']).reset_index()
                #squared_stats = df_w.copy(deep = True)
                #print(squared_stats)
                ##exit()
                #squared_stats = squared_stats.pow[metric_key] ** 2
                #squared_stats = squared_stats.groupby('q')[metric_key].agg(['mean']).reset_index()
                N = math.pow(width,2)

                chi_df = (
                    df_w.groupby("q")[metric_key].agg(
                        mean_s="mean",
                        mean_s2=lambda x: np.square(x).mean()
                    ).reset_index()
                )


                chi_df["chi"] = N * (chi_df["mean_s2"]-chi_df["mean_s"]**2)
                print(chi_df)
                number_realisations = df_w.groupby('q')[metric_key].size().iloc[0]
                #print(number_realisations)
                #exit()
                # Plot line with error bars (standard deviation)

                plt.plot(chi_df["q"], chi_df["chi"], marker="o",  label=f"N = {width}²")


            # Labels and Aesthetics
            plt.title(f"Axelrod Model, regular lattice, χ distribution of {metric_key}, F = {F}", pad=20)
            plt.xlabel("Number of Traits per Feature (q)")
            plt.ylabel(f"χ({metric_key})")
            #plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(title="Grid Size", frameon=True)
            
            # Save Logic
            file_name =  f"axelrod_F{F}_{metric_key}_Chi.png"
            save_path = plot_dir / file_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() # Close figure to free memory

    print(f"Update complete. Plots saved to: {plot_dir}")

    for F in unique_Fs:
        df_f = df[df['F'] == F]
        
        for metric_key, metric_name in metrics:
            plt.figure(figsize=(10, 7))
            
            # Group by width (N) and q to get statistics
            # We use width for the legend label
            widths = sorted(df_f['width'].unique())
            
            #for width in widths:
            for width in ploted_widths:
            #to recreate graph from paper on these widths
                df_w = df_f[df_f['width'] == width]
                
                # Aggregate data for each q
                stats = df_w.groupby('q')[metric_key].agg(['mean', 'std']).reset_index()
                #squared_stats = df_w.copy(deep = True)
                #print(squared_stats)
                ##exit()
                #squared_stats = squared_stats.pow[metric_key] ** 2
                #squared_stats = squared_stats.groupby('q')[metric_key].agg(['mean']).reset_index()
                N = math.pow(width,2)

                chi_df = (
                    df_w.groupby("q")[metric_key].agg(
                        mean_s="mean",
                        mean_s2=lambda x: np.square(x).mean()
                    ).reset_index()
                )

                #VAR
                chi_df["chi"] =  (chi_df["mean_s2"]-chi_df["mean_s"]**2)
                print(chi_df)
                number_realisations = df_w.groupby('q')[metric_key].size().iloc[0]
                #print(number_realisations)
                #exit()
                # Plot line with error bars (standard deviation)

                plt.plot(chi_df["q"], chi_df["chi"], marker="o",  label=f"N = {width}²")


            # Labels and Aesthetics
            plt.title(f"Axelrod Model, regular lattice, χ distribution of {metric_key}, F = {F}", pad=20)
            plt.xlabel("Number of Traits per Feature (q)")
            plt.ylabel(f"χ({metric_key})")
            #plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(title="Grid Size", frameon=True)
            
            # Save Logic
            file_name =  f"axelrod_F{F}_{metric_key}_VAR.png"
            save_path = plot_dir / file_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() # Close figure to free memory

    for F in unique_Fs:
        df_f = df[df['F'] == F]
        for q0 in range(13,16):
            for metric_key, metric_name in metrics:
                plt.figure(figsize=(10, 7))
                
                # Group by width (N) and q to get statistics
                # We use width for the legend label
                widths = sorted(df_f['width'].unique())
                
                #for width in widths:
                #to recreate graph from paper on these widths
                df_w = df_f[df_f['width'] == 100]
                data = df_w.loc[df["q"] == q0, metric_key]
                    
                    # Aggregate data for each q
                    #squared_stats = df_w.copy(deep = True)
                    #print(squared_stats)
                    ##exit()
                    #squared_stats = squared_stats.pow[metric_key] ** 2
                    #squared_stats = squared_stats.groupby('q')[metric_key].agg(['mean']).reset_index()
                N = math.pow(width,2)

                    #print(number_realisations)
                    #exit()
                    # Plot line with error bars (standard deviation)

                plt.hist(data, bins=30)


                # Labels and Aesthetics
                plt.title(f"Axelrod Model, regular lattice, Histogram of {metric_key} for q = {q0}, M=500 realizations, F = {F}", pad=20)
                plt.xlabel(f"{metric_key} distribution")
                plt.ylabel("Number of iterations")
                #plt.ylim(-0.05, 1.05) # Values are normalized 0 to 1
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Save Logic
                file_name =  f"axelrod_F{F}_{metric_key}_q{q0}_Hist.png"
                save_path = plot_dir / file_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close() # Close figure to free memory

    print(f"Update complete. Plots saved to: {plot_dir}")

if __name__ == "__main__":
    plot_axelrod_data()