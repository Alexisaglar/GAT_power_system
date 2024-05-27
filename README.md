# Power Network GAT

This project uses a Graph Attention Network (GAT) to predict voltage magnitude and angle in a power network.

## Setup

1. Clone the repository:
    ```
    git clone https://github.com/alexisaglar/GAT_power_system.git
    ```

2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Place your data in the `data/raw` directory and run the preprocessing script (if needed).

## Usage

1. Train the model:
    ```
    python src/main.py
    ```

2. Evaluate the model:
    ```
    python src/evaluate.py
    ```

## Project Structure

- `raw_data/`: Contains raw and processed data.
- `src/`: Contains source code for data loading, model definition, training, and evaluation.
- `plots/`: Contains plots generated during network generation and results.
- `requirements.txt`: Lists the dependencies needed to run the project.
- `README.md`: Provides an overview and instructions for the project.
