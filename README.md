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
    python src/train.py
    ```

2. Evaluate the model:
    ```
    python src/evaluate.py
    ```

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Contains Jupyter notebooks for data exploration.
- `src/`: Contains source code for data loading, model definition, training, and evaluation.
- `plots/`: Contains plots generated during data exploration or training.
- `requirements.txt`: Lists the dependencies needed to run the project.
- `README.md`: Provides an overview and instructions for the project.
