## Requirements
- Python 3.8+
- PyTorch 1.13+
- pandas, numpy, scikit-learn

## Data Preparation
Due to patient privacy and institutional regulations, the full WSI and genomic datasets cannot be publicly shared at this stage. However, we provide a dummy dataset to verify the execution of our code.
Please place the generated dummy data in the `./data/` directory.

## Usage

### 1. Training
To train the MAKE model using 5-fold cross-validation, run:
```bash
python train.py --fold 0 --use_kan 1 --w_pc1 0.1