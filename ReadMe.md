# vanguard-ab-test

Project presented by Benjamin Lambour (Github: BenjaminIH) and Diego Llorente (Github: Diego-Llorente).

The presentation can be followed on this webpage:
https://docs.google.com/presentation/d/1v4PXOTszJyMnlQf9eNxT0DIidPwvZqQOtCplw3YrUDA/edit?usp=sharing

The data used for this project can be found using this link:
https://github.com/data-bootcamp-v4/lessons/tree/main/5_6_eda_inf_stats_tableau/project/files_for_project

## Overview

This project focuses on analyzing an A/B test conducted to evaluate the effectiveness of a new design compared to an old design. We utilized various CSV files as our data sources, cleaned and analyzed the data using Jupyter Notebook, and created visualizations with Tableau to present our findings.

## Table of Contents

- [Introduction](#introduction)
- [Data Sources](#data-sources)
- [Data Cleaning](#data-cleaning)
- [Analysis](#analysis)
- [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
- [Visualizations](#visualizations)
- [Experiment Evaluation](#experiment-evaluation)
- [Conclusion](#conclusion)
- [Future Recommendations](#future-recommendations)

## Introduction

The goal of this analysis is to assess the design effectiveness of the A/B test, focusing on whether the experiment was well-structured, the randomization of clients, potential biases, and the adequacy of the experiment's timeframe.

## Data Sources

- Various CSV files containing user data and interaction metrics.
- Data includes metrics like group size, completion rates, client age, and tenure.

## Data Cleaning

The initial data underwent several cleaning steps:
- Removed duplicates and irrelevant entries.
- Handled missing values.
- Converted data types for accurate analysis.

## Analysis

We performed statistical analyses to compare the control and test groups across several metrics:

- **Group Sizes**:
  - Test Group: n_test = 26,961
  - Control Group: n_control = 23,526
  
- **Completion Rates**:
  - Significantly different with a p-value = 1.03239e−15
  
- **Average Client Age**:
  - Significantly different with a p-value = 0.0281577203
  
- **Client Tenure**:
  - Statistically similar with a p-value = 0.0281

## Key Performance Indicators (KPIs)

We calculated the following KPIs to assess the performance of both designs:

- **Completion Rate**: The percentage of users who completed the desired action.
- **Error Rate**: The percentage of users who encountered errors during their interaction.
- **Step Time**: The average time taken by users to complete each step in the process.

These KPIs provided valuable insights into user experience and interaction efficiency between the test and control groups.

## Visualizations

Visualizations were created using Tableau to depict the differences between the two groups and highlight key findings in a more digestible format.

## Experiment Evaluation

### Design Effectiveness

- **Structure**: The experiment was not well-structured for a correct comparison.
  
- **Randomization**: Clients were not randomly and equally divided, leading to biases:
  - Significant differences in completion rates and client age suggest issues with randomization.
  
- **Timeframe**: The experiment's duration (3/15/2017 to 6/20/2017) was adequate, but the flawed design impacted validity.

## Conclusion

The analysis revealed significant biases and structural issues in the A/B test, compromising the validity of the results. While the timeframe was sufficient for data collection, the underlying problems suggest that insights drawn from this test may not be reliable.

## Future Recommendations

- Ensure proper randomization in future A/B tests to avoid biases.
- Control for variables that could affect group comparability.
- Consider additional metrics to evaluate the overall effectiveness of design changes.

## Getting Started

To replicate this analysis or explore further, clone the repository and ensure you have Jupyter Notebook and Tableau installed.

### Requirements

- Python 3.x
- Libraries: pandas, numpy, scipy, matplotlib, seaborn
- Tableau for visualizations



