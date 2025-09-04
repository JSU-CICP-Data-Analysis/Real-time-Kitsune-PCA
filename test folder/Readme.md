# Purpose of the 'test' Folder

This folder serves as a **development and testing environment** for a network anomaly detection pipeline based on the **Kitsune** algorithm. The files within it are designed to demonstrate and evaluate different stages of the workflow, from data handling to analysis and visualization.

### Key Functions

* **PCAP Data Preparation**: Scripts for processing and slicing large PCAP files (`pcap_analyzer_(test).py`, `slicing_pcap.ipynb`) to create manageable datasets for analysis.
* **Parallel Processing**: Implementations for running multiple instances of the Kitsune algorithm in parallel (`parallel_kitsune_(test).py`), which is a key method for improving performance on large datasets.
* **Anomaly Detection & Results Saving**: The core Kitsune-based detection logic is integrated and tested (`pcap_attack_analyzer_(test).py`). The `save_results_(test).py` script handles the structured output of anomaly scores and confidence metrics.
* **Data Visualization**: Scripts for generating plots (`plot(time)_(test).py`, `plotting (packet index)_(test).py`) that visualize the anomaly scores (RMSE) over time or by packet index.
* **Performance Evaluation**: A Jupyter notebook (`rmse_(test).ipynb`) is included to evaluate the model's effectiveness by calculating key metrics like **Precision, Recall, and F1-Score** against known attack data.
