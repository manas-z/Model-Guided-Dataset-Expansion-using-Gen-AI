# Model-Guided Dataset Expansion using Gen AI

This repository contains a deep learning mini-project on targeted synthetic data augmentation for imbalanced chest X-ray classification. The project uses a pretrained ResNet18 classifier to identify the weakest minority class, trains a DCGAN on that class, and adds generated samples back into the training set before fine-tuning the classifier.

The experiment is built around the COVID-19 Radiography Dataset and compares model-guided GAN augmentation against continued training and simple real-image oversampling.

## Project Idea

Medical imaging datasets are often imbalanced: common classes have many examples, while clinically important disease classes may have fewer samples. Instead of blindly augmenting every class, this project uses the classifier's validation recall to decide which minority class needs more representation.

The pipeline is:

1. Load and explore the COVID-19 chest X-ray dataset.
2. Train a baseline ResNet18 classifier.
3. Identify the weakest minority class using validation recall.
4. Train a DCGAN only on that selected class.
5. Generate synthetic images for the weak class.
6. Fine-tune the classifier with the expanded dataset.
7. Compare against continued training and real-image oversampling controls.

## Repository Contents

| File | Description |
| --- | --- |
| `main_eda.ipynb` | Exploratory data analysis, class distribution checks, and image visualizations. |
| `main.ipynb` | Main training pipeline: ResNet18 baseline, weak-class selection, DCGAN training, augmentation, and final evaluation. |
| `.gitignore` | Ignores local datasets, virtual environments, checkpoints, and cache files. |

## Dataset

This project uses the COVID-19 Radiography Dataset with four classes:

- COVID
- Lung Opacity
- Normal
- Viral Pneumonia

Dataset size used in the notebook:

| Class | Images |
| --- | ---: |
| COVID | 3,616 |
| Lung Opacity | 6,012 |
| Normal | 10,192 |
| Viral Pneumonia | 1,345 |
| **Total** | **21,165** |

The notebooks expect the dataset directory to contain class folders with an `images/` subfolder, for example:

```text
COVID-19_Radiography_Dataset/
  COVID/images/
  Lung_Opacity/images/
  Normal/images/
  Viral Pneumonia/images/
```

Update the dataset path inside both notebooks before running:

```python
DATA_DIR = r"path\to\COVID-19_Radiography_Dataset"
DATASET_PATH = r"path\to\COVID-19_Radiography_Dataset"
```

## Methodology

### Baseline Classifier

- Backbone: pretrained ResNet18
- Frozen convolutional layers
- Custom classifier head with dropout
- Loss: weighted cross-entropy
- Optimizer: Adam
- Input size: 224 x 224

### Weak-Class Selection

After baseline training, the model computes validation recall for the minority classes. In the saved run, the minority candidates were:

- Viral Pneumonia
- COVID

COVID had lower validation recall among these candidates, so it was selected for synthetic expansion.

### DCGAN Augmentation

- Generator input: 100-dimensional latent vector
- Output image size: 64 x 64 RGB
- Training class: COVID
- Training images: 2,893 COVID training images
- Epochs: 60
- Synthetic images generated: 800

The generated images are resized to 224 x 224 and added to the classifier training set.

## Results

Final evaluation was performed on a held-out test set of 2,117 images.

| Method | Accuracy | Macro F1 | COVID Recall |
| --- | ---: | ---: | ---: |
| Baseline | 87.10% | 0.8724 | 88.40% |
| Continue only | 87.58% | 0.8773 | 91.16% |
| Real oversample | 87.77% | 0.8812 | 92.82% |
| GAN augmented | **87.86%** | **0.8833** | 90.61% |

The GAN-augmented model achieved the best overall accuracy and macro F1 score. Real oversampling produced the highest COVID recall, but GAN augmentation gave the best overall class-balanced performance.

## Setup

Create a Python environment and install the required packages:

```bash
pip install numpy pandas matplotlib seaborn pillow opencv-python tqdm scikit-learn torch torchvision
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation guide.

## How to Run

1. Download and extract the COVID-19 Radiography Dataset.
2. Update `DATASET_PATH` in `main_eda.ipynb`.
3. Run `main_eda.ipynb` to verify the dataset structure and class distribution.
4. Update `DATA_DIR` in `main.ipynb`.
5. Run `main.ipynb` from top to bottom.

The main notebook will train the baseline classifier, identify the weak class, train the DCGAN, generate synthetic images, and compare all evaluation settings.

## Key Takeaways

- Targeted data augmentation can be more useful than uniformly augmenting all classes.
- Validation recall is a simple but practical signal for deciding where augmentation is needed.
- GAN-based augmentation gave a modest improvement over continued training and real-image oversampling in this experiment.
- The improvement is measurable but not large, so stronger validation would be needed before making broader claims.

## Limitations

- The experiment uses one dataset and one train-validation-test split.
- The DCGAN generates low-resolution 64 x 64 images.
- The ResNet18 backbone remains frozen during fine-tuning.
- No external clinical validation is performed.
- The project is for educational and research purposes only and is not intended for medical diagnosis.

## Future Work

- Repeat experiments across multiple random seeds.
- Add external dataset validation.
- Try conditional GANs, diffusion models, or higher-resolution generators.
- Compare against stronger imbalance methods such as focal loss and class-balanced loss.
- Use uncertainty, calibration error, or sample-level loss for more precise augmentation targeting.

## Disclaimer

This project is a research and educational prototype. It should not be used for clinical decision-making or medical diagnosis.
