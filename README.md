# Inventory_Optimization_Model
Calculated optimal Reorder Points (ROP) and Safety Stock (SS) using Python/Pandas, demonstrating data segmentation to mitigate promotional demand bias.

Inventory Reorder Point (ROP) Optimization Model

🎯 Overview

This project implements a quantitative inventory management solution to establish optimized Reorder Points (ROP) and Safety Stock (SS) for critical SKUs. The model utilizes the Normal Distribution and a target 95% Service Level ($Z = 1.64$) to balance service reliability against working capital investment.

The core success of this project was identifying and mitigating a critical data bias, leading to a deeper understanding of underlying demand risk.

🛠️ Technology Stack

Data Preparation: SQL (Data Transformation, Aggregation)

Modeling & Analysis: Python (Pandas, NumPy, SciPy)

Visualization: Matplotlib

Documentation: GitHub

🔑 Key Analytical Insight: Baseline Volatility

The initial hypothesis was that frequent promotions (50% of operating days) were the primary source of inventory risk. The model was designed to test this by comparing two scenarios:

ALL Demand: Unfiltered data, including promotional spikes.

NORMAL Demand: Data filtered to exclude promotions.

The analysis revealed that the Standard Deviation of Daily Demand ($\sigma_D$), the key driver of Safety Stock, was nearly identical in both scenarios:

Scenario

Average Daily Demand ($\bar{D}$)

Standard Deviation ($\sigma_D$)

Reorder Point (ROP)

ALL Demand (Initial)

102.01

39.17

999

NORMAL Demand (Filtered)

100.23

39.899

988

Conclusion: The negligible difference in $\sigma_D$ proves that the high Safety Stock requirement is not due to promotions (which are predictable), but to an extreme, inherent volatility in baseline demand. The high ROP is a necessity given the current high level of unpredictable daily fluctuation.

💡 Strategic Recommendations

The project concludes with three actionable steps for management:

Immediate Adoption: Implement the ROP values from the NORMAL Demand model to establish data integrity and ensure routine purchasing buffers against unpredictable risk only.

Phase II Investigation: Launch an in-depth analysis of the 'NORMAL' demand data to identify the root causes of the high baseline volatility ($\sigma_D \approx 40$), focusing on seasonality, competitor activity, or distribution inconsistencies.

Cost Prioritization: Use the calculated Cost of Safety Stock metric to prioritize risk mitigation efforts on the most expensive and volatile SKUs.

📈 Visualizing the High ROP

The following chart illustrates the high daily volatility and the resulting ROP level required to maintain the 95% service target over the vendor lead time.

[View ROP Plot All](visuals/ROP_plot_SKU_18_ALL.png)

[View ROP Plot Normal](visuals/ROP_plot_SKU_18_NORMAL.png)

📁 Repository Structure

/code: Source code (inventory_model.py)

/data_samples: Sample input data (CSV files)

/docs: Final business-facing report (executive_summary.pdf)

/output: Raw model output (final CSV recommendations)

/visuals: Project visualizations (PNGs)