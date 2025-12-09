import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# --- 1. CONFIGURATION: UPDATE THESE VALUES ---
# File exported from SQL query (Date, SKU, Daily_Demand, is_promotion)
INPUT_DEMAND_PATH = 'aggregated_daily_demand.csv'
# File exported from SQL query (SKU, Lead_Time_Days, Unit_Cost)
INPUT_MASTER_PATH = 'item_master_data.csv'

# The service level we aim for (e.g., 95% = 5% chance of stockout)
SERVICE_LEVEL = 0.95
Z_SCORE = norm.ppf(SERVICE_LEVEL)

# *** NEW CONFIGURATION FOR DEMAND SCENARIO ***
# Options: 'NORMAL' (removes promotional demand) or 'ALL' (uses all demand)
DEMAND_SCENARIO = 'NORMAL'

# Configuration for Visualization
SKU_FOR_VISUALIZATION = 'SKU_18'

print(f"--- Inventory Optimization Model Configuration ---")
print(f"Target Service Level: {SERVICE_LEVEL*100:.0f}% (Z-Score: {Z_SCORE:.3f})")
print(f"Demand Scenario: {DEMAND_SCENARIO}")
print(f"SKU Selected for Detail Plot: {SKU_FOR_VISUALIZATION}")
print(f"--------------------------------------------------")

# --- 2. Load Data and Define Inputs ---
try:
    # Load Aggregated Daily Demand Data
    df = pd.read_csv(INPUT_DEMAND_PATH)
    # Ensure the date column is sorted and the SKU_ID is consistent
    df['day_of_demand'] = pd.to_datetime(df['day_of_demand']) # Convert date column
    df = df.sort_values(by=['sku', 'day_of_demand'])

    # *** DEMAND FILTERING LOGIC ***
    if DEMAND_SCENARIO == 'NORMAL' and 'is_promotion' in df.columns:
        original_count = len(df)
        df = df[df['is_promotion'] == 0].copy() # Filter out promotion days
        print(f"Filtered {original_count - len(df)} promotional demand days. Using {len(df)} records for 'NORMAL' modeling.")

    elif DEMAND_SCENARIO == 'NORMAL' and 'is_promotion' not in df.columns:
        print(f"WARNING: DEMAND_SCENARIO is 'NORMAL' but 'is_promotion' column not found. Using ALL demand.")

    print(f"Loaded {len(df)} daily demand records.")

except FileNotFoundError:
    print(f"Error: Daily Demand file not found at {INPUT_DEMAND_PATH}. Please check the file name.")
    exit()

try:
    # Load Item Master Data (Static Attributes)
    item_master_df = pd.read_csv(INPUT_MASTER_PATH)
    item_master_df = item_master_df.rename(columns={'SKU_ID': 'sku'}) # Ensure merge column is consistent
    # Ensure the Lead Time and Unit Cost columns exist and are numeric
    item_master_df['Lead_Time_Days'] = pd.to_numeric(item_master_df['Lead_Time_Days'], errors='coerce')
    item_master_df['Unit_Cost'] = pd.to_numeric(item_master_df['Unit_Cost'], errors='coerce')
    print(f"Loaded {len(item_master_df)} unique item master records.")
except FileNotFoundError:
    print(f"Error: Item Master file not found at {INPUT_MASTER_PATH}. Please check the file name.")
    exit()

# --- 3. Statistical Calculation (Groupby) ---
# Group the clean daily demand data by SKU to calculate the key metrics:
metrics_df = df.groupby('sku')['Daily_Demand'].agg(
    Avg_Daily_Demand = 'mean',            # D̄
    Std_Dev_Daily_Demand = 'std',         # σD
    Historical_Days = 'count'             # Validate data volume
).reset_index()

# Merge the calculated metrics with the item master data
model_df = pd.merge(metrics_df, item_master_df, on='sku', how='left')

# Handle SKUs with missing master data
model_df.dropna(subset=['Lead_Time_Days', 'Unit_Cost'], inplace=True)
print(f"Analysis proceeding with {len(model_df)} SKUs after merging master data.")

# --- 4. ROP and Safety Stock Formulas ---
# Check for any zero or negative lead times to prevent errors in np.sqrt
model_df = model_df[model_df['Lead_Time_Days'] > 0].copy()

# Calculate Safety Stock (SS)
# SS = Z * σD * sqrt(L)
model_df['Safety_Stock'] = (
    Z_SCORE * model_df['Std_Dev_Daily_Demand'] * np.sqrt(model_df['Lead_Time_Days'])
)

# Calculate Reorder Point (ROP)
model_df['Reorder_Point_ROP'] = (
    (model_df['Avg_Daily_Demand'] * model_df['Lead_Time_Days']) +
    model_df['Safety_Stock']
)

# Calculate Cost of Safety Stock (for Executive Summary)
# Cost_of_SS = SS * Unit_Cost
model_df['Cost_of_Safety_Stock'] = model_df['Safety_Stock'] * model_df['Unit_Cost']

# Round the final inventory numbers to whole units
model_df['Safety_Stock'] = np.ceil(model_df['Safety_Stock']).astype(int)
model_df['Reorder_Point_ROP'] = np.ceil(model_df['Reorder_Point_ROP']).astype(int)


# --- 5. Final Output and Documentation ---
FINAL_OUTPUT_FILE = 'inventory_recommendations_model.csv'
model_df.to_csv(FINAL_OUTPUT_FILE, index=False)

print("\n--- Final Inventory Recommendations (First 5 SKUs) ---")
print("These values represent the new strategic inventory levels.")
print(model_df[['sku', 'Avg_Daily_Demand', 'Lead_Time_Days',
                'Safety_Stock', 'Reorder_Point_ROP', 'Cost_of_Safety_Stock']].head().to_markdown(index=False))

print(f"\nModeling complete! Recommendations saved to '{FINAL_OUTPUT_FILE}'")


# --- 6. Visualization ---

# 6.1. Demand Variability and ROP Plot
def plot_demand_variability(sku_id, demand_df, model_results):
    """Generates a time-series plot of demand vs. ROP for a single SKU."""
    sku_data = demand_df[demand_df['sku'] == sku_id]

    # Check if SKU exists in final model results
    sku_metrics_row = model_results[model_results['sku'] == sku_id]
    if sku_metrics_row.empty or sku_data.empty:
        print(f"Warning: Data not found for SKU {sku_id}. Skipping plot.")
        return

    sku_metrics = sku_metrics_row.iloc[0]
    rop = sku_metrics['Reorder_Point_ROP']
    avg_demand = sku_metrics['Avg_Daily_Demand']

    plt.figure(figsize=(12, 6))
    plt.plot(sku_data['day_of_demand'], sku_data['Daily_Demand'], label='Historical Daily Demand', color='darkslategray', alpha=0.7)

    # Add ROP and Average Demand Lines
    plt.axhline(y=rop, color='red', linestyle='--', linewidth=2, label=f'Reorder Point (ROP): {rop} Units')
    plt.axhline(y=avg_demand, color='blue', linestyle=':', linewidth=1.5, label=f'Average Daily Demand: {avg_demand:.2f} Units')

    plt.title(f'Demand Variability and Reorder Point for SKU: {sku_id} (Scenario: {DEMAND_SCENARIO})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Demand Units', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # *** Use Matplotlib Locators and Formatters to declutter X-axis ***
    plt.gca().xaxis.set_major_locator(MonthLocator(interval=1)) # Show major tick every 1 month
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m')) # Format date as Year-Month
    plt.xticks(rotation=45) # Rotate for better readability

    plt.tight_layout()
    plt.savefig(f'Demand_ROP_Plot_{sku_id}_{DEMAND_SCENARIO}.png')
    plt.close()
    print(f"Visualization saved: Demand_ROP_Plot_{sku_id}_{DEMAND_SCENARIO}.png")

# 6.2. Inventory Cost vs. Demand Variability Plot (Scatter Plot)
def plot_cost_risk(model_results):
    """Generates a scatter plot showing the financial trade-off for all SKUs."""
    plt.figure(figsize=(10, 8))

    # Scatter plot: Cost of Safety Stock (Y-axis) vs. Std Dev of Demand (X-axis)
    plt.scatter(model_results['Std_Dev_Daily_Demand'], model_results['Cost_of_Safety_Stock'],
                alpha=0.6, edgecolors='w', s=model_results['Avg_Daily_Demand']*50, # Size by Avg Demand
                label='SKUs (Size based on Avg Daily Demand)')

    # Label a few high-cost/high-risk SKUs
    for i, row in model_results.sort_values('Cost_of_Safety_Stock', ascending=False).head(3).iterrows():
        plt.annotate(row['sku'], (row['Std_Dev_Daily_Demand'] * 1.05, row['Cost_of_Safety_Stock']),
                     fontsize=9, color='red')

    plt.title(f'Inventory Cost vs. Demand Variability (Scenario: {DEMAND_SCENARIO})', fontsize=16)
    plt.xlabel('Standard Deviation of Daily Demand (Demand Risk, $\sigma_D$)', fontsize=12)
    plt.ylabel(f'Cost of Safety Stock (Inventory Investment, @ {SERVICE_LEVEL*100:.0f}% SL)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Cost_Risk_Scatter_Plot_{DEMAND_SCENARIO}.png')
    plt.close()
    print(f"Visualization saved: Cost_Risk_Scatter_Plot_{DEMAND_SCENARIO}.png")

# Execute Visualization Functions
plot_demand_variability(SKU_FOR_VISUALIZATION, df, model_df)
plot_cost_risk(model_df)

print(f"--- Visualization Complete ---")