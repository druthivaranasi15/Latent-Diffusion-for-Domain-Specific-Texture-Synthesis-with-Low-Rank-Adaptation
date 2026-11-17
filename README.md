# Latent Diffusion for Domain-Specific Texture Synthesis with LoRA

## Project Overview

This project explores domain-specific synthetic image generation using **Stable Diffusion 1.5** and **Low-Rank Adaptation (LoRA)**. The experiments use tomato leaf disease images from the PlantVillage dataset to investigate whether fine-tuning a pretrained diffusion model can generate realistic domain-specific images and whether those synthetic images can support a downstream image-classification task.

Three configurations are considered:

1. **Baseline:** Pretrained Stable Diffusion 1.5 without domain-specific LoRA fine-tuning.
2. **LoRA Rank 8:** Stable Diffusion 1.5 fine-tuned using a LoRA adapter with rank 8.
3. **LoRA Rank 16:** Stable Diffusion 1.5 fine-tuned using a LoRA adapter with rank 16.

The generated images are evaluated visually and through a downstream classification experiment using a pretrained ResNet-18 model.

The main objective is to compare the effect of LoRA fine-tuning and adapter rank on domain-specific image generation and the usefulness of synthetic images for image classification.

---

## Table of Contents

- [Objectives](#objectives)
- [Dataset and Input Details](#dataset-and-input-details)
- [Project Architecture](#project-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Generative Model and LoRA Fine-Tuning](#generative-model-and-lora-fine-tuning)
- [Synthetic Image Generation](#synthetic-image-generation)
- [Downstream Classification](#downstream-classification)
- [Experimental Results](#experimental-results)
- [Visualizations](#visualizations)
- [Comparison and Ablation Study](#comparison-and-ablation-study)
- [Reproducibility and Evaluation Notes](#reproducibility-and-evaluation-notes)
- [Repository Structure](#repository-structure)
- [Execution Instructions](#execution-instructions)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Objectives

The project has the following objectives:

- Explore Stable Diffusion 1.5 for domain-specific image generation.
- Fine-tune the pretrained diffusion model using LoRA.
- Compare two LoRA adapter configurations: Rank 8 and Rank 16.
- Generate synthetic tomato leaf images across four disease categories.
- Evaluate the generated data using a downstream image-classification task.
- Compare classification performance and training time across the reported configurations.
- Study how changing the LoRA rank affects adapter size and experimental performance.

---

## Dataset and Input Details

### Dataset

The project uses the **PlantVillage dataset**, obtained through the Kaggle dataset source:

[PlantVillage Dataset – Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

The original dataset contains images from multiple plant types and disease categories. For this project, four tomato leaf categories were selected.

### Selected classes

| Class ID | Class name | Description |
|---:|---|---|
| 0 | Tomato_Bacterial_spot | Tomato leaves affected by bacterial spot |
| 1 | Tomato_Early_blight | Tomato leaves affected by early blight |
| 2 | Tomato_Late_blight | Tomato leaves affected by late blight |
| 3 | Tomato_healthy | Healthy tomato leaves |

### Dataset statistics

The four selected categories initially contained 6,627 images.

After image verification and exact duplicate removal, the final dataset contained **6,613 valid images**.

| Dataset split | Number of images | Percentage |
|---|---:|---:|
| Training | 4,629 | 70% |
| Validation | 992 | 15% |
| Testing | 992 | 15% |
| **Total** | **6,613** | **100%** |

### Class-wise dataset distribution

| Class | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Tomato_Bacterial_spot | 1,489 | 319 | 319 | 2,127 |
| Tomato_Early_blight | 700 | 150 | 150 | 1,000 |
| Tomato_Late_blight | 1,331 | 285 | 285 | 1,901 |
| Tomato_healthy | 1,109 | 238 | 238 | 1,585 |
| **Total** | **4,629** | **992** | **992** | **6,613** |

**Note:** The class-wise counts above should be checked against the final notebook manifest before publication. The notebook's initial class counts and the final class-wise split output should be treated as the authoritative source if any discrepancy appears.

### Input format

- Input type: Leaf images
- Supported image extensions in the preprocessing code: `.jpg`, `.jpeg`, `.png`
- Generative model training resolution: 512 × 512 pixels
- Downstream classifier input resolution: 224 × 224 pixels
- Image color representation: RGB

---

## Project Architecture

The workflow consists of five main stages:

1. Dataset preparation and preprocessing.
2. Baseline synthetic image generation using pretrained Stable Diffusion 1.5.
3. Domain-specific fine-tuning using LoRA Rank 8 and Rank 16.
4. Synthetic image generation using the fine-tuned models.
5. Downstream image classification and experimental comparison.

```text
                 PlantVillage Dataset
                           |
                           v
               Select Four Tomato Classes
                           |
                           v
              Image Verification and Cleaning
                           |
                           v
                Duplicate Removal (SHA-256)
                           |
                           v
              Stratified Dataset Splitting
                           |
              +------------+-------------+
              |            |             |
              v            v             v
            Train       Validation      Test
              |
              v
       Text Caption Preparation
              |
              v
      Stable Diffusion 1.5
              |
       +------+------+
       |             |
       v             v
    LoRA Rank 8   LoRA Rank 16
       |             |
       v             v
    Synthetic     Synthetic
     Images        Images
       |             |
       +------+------+
              |
              v
       ResNet-18 Classification
              |
              v
       Performance Comparison
```

---

## Data Preprocessing

Preprocessing was performed to improve dataset quality, avoid duplicate samples, and maintain a consistent experimental setup.

### 1. Class selection

Four tomato leaf categories were selected from the PlantVillage dataset.

This reduces the scope of the experiment and makes it possible to compare generated images across specific disease categories.

### 2. Image verification

The selected image files were checked to ensure they could be opened and processed.

The notebook reports:

- Images scanned: 6,627
- Valid images: 6,627
- Corrupted images: 0

This verification prevents invalid image files from entering the training pipeline.

### 3. Duplicate detection and removal

SHA-256 hashing was used to identify exact duplicate images.

The notebook reports:

- Exact duplicate groups: 14
- Images involved in duplicate groups: 28
- Exact duplicates removed: 14
- Final cleaned images: 6,613

Removing duplicates reduces redundant samples and helps prevent identical images from appearing in different dataset splits.

### 4. Stratified train-validation-test splitting

The cleaned dataset was divided into:

- 70% training
- 15% validation
- 15% testing

Stratified splitting maintains the class distribution across the three subsets.

A fixed random seed was used to support reproducibility.

### 5. Duplicate leakage verification

The image hashes were checked across the training, validation, and testing subsets.

The notebook reports zero duplicate hashes appearing across multiple splits.

This check helps ensure that exact duplicate images do not leak between the subsets.

### 6. Text caption preparation

Text captions were created for the selected tomato leaf categories.

Examples include:

- A close-up photograph of a tomato leaf affected by bacterial spot disease.
- A close-up photograph of a tomato leaf affected by late blight disease.
- A close-up photograph of a healthy green tomato leaf.

These captions provide text conditioning for the Stable Diffusion training process.

### 7. Image processing for diffusion training

The LoRA training configuration uses:

- Image resolution: 512 × 512
- Center cropping
- Random horizontal flipping

These operations provide consistent image dimensions and introduce limited variation during training.

### 8. Image processing for classification

The downstream ResNet-18 classifier uses:

- Resizing to 224 × 224
- Random horizontal flipping during training
- Random rotation of up to 10 degrees during training
- Conversion to tensors
- ImageNet normalization

ImageNet normalization is used because the classifier starts from pretrained ResNet-18 weights.

---

## Generative Model and LoRA Fine-Tuning

### Base model

The project uses **Stable Diffusion 1.5**.

Model identifier:

`stable-diffusion-v1-5/stable-diffusion-v1-5`

The same base model and model revision are used for the LoRA experiments.

### What is LoRA?

Low-Rank Adaptation is a parameter-efficient fine-tuning method.

Instead of updating all the original model parameters, LoRA introduces smaller trainable adapter matrices into selected model layers.

This reduces the number of parameters that need to be trained and allows domain-specific adaptation while retaining the pretrained diffusion model.

### Experimental configurations

| Configuration | Description |
|---|---|
| Baseline | Pretrained Stable Diffusion 1.5 |
| LoRA Rank 8 | Stable Diffusion 1.5 with Rank 8 adapters |
| LoRA Rank 16 | Stable Diffusion 1.5 with Rank 16 adapters |

### LoRA training configuration

The following configuration is used for the fine-tuning experiments.

| Parameter | Value |
|---|---:|
| Base model | Stable Diffusion 1.5 |
| Training images | 4,629 |
| Training resolution | 512 × 512 |
| LoRA Rank | 8 or 16 |
| Maximum training steps | 3,500 |
| Training epochs | 4 |
| Batch size per device | 1 |
| Gradient accumulation steps | 4 |
| Effective batch size | 4 |
| Learning rate | 1e-4 |
| Learning-rate scheduler | Cosine |
| Warmup steps | 100 |
| Maximum gradient norm | 1.0 |
| Mixed precision | FP16 |
| Random seed | 42 |
| Gradient checkpointing | Enabled |
| Checkpoint interval | 500 steps |

The main experimental variable is the LoRA rank. The remaining training settings are kept consistent to support a controlled comparison.

### Adapter size

The saved adapters have the following reported sizes:

| Adapter | Approximate size |
|---|---:|
| LoRA Rank 8 | 6.12 MB |
| LoRA Rank 16 | 12.20 MB |

The Rank 16 adapter is larger because it has a higher adapter rank and therefore greater adapter parameter capacity.

---

## Synthetic Image Generation

After fine-tuning, the trained LoRA adapters are loaded into Stable Diffusion 1.5 to generate domain-specific tomato leaf images.

### Generation configuration

| Parameter | Value |
|---|---:|
| Number of classes | 4 |
| Images per class | 100 |
| Expected images per experiment | 400 |
| Image resolution | 512 × 512 |
| Inference steps | 25 |
| Guidance scale | 7.5 |

The same disease-specific prompts are used for the generation experiments to support comparison.

### Generation quality checks

The generation pipeline includes checks for:

- Missing image files
- Black images returned by the safety checker
- Generation failures
- Successful image counts per class

The Rank 8 generation output reports:

| Class | Valid generated images |
|---|---:|
| Tomato_Bacterial_spot | 100 |
| Tomato_Early_blight | 100 |
| Tomato_Late_blight | 100 |
| Tomato_healthy | 100 |
| **Total** | **400** |

The Rank 8 validation output reports zero black images and zero missing images.

The Rank 16 generation output should be checked using its saved generation metadata and validation output before reporting its final valid-image count.

---

## Downstream Classification

To investigate whether synthetic images are useful for a classification task, a pretrained **ResNet-18** model is used.

The classifier is trained on real images combined with generated synthetic images.

### Training data

For the baseline and Rank 8 classifier experiments, the notebook verifies:

| Data | Number of images |
|---|---:|
| Real training images | 4,629 |
| Synthetic training images | 400 |
| Combined training images | 5,029 |
| Real validation images | 992 |
| Real test images | 992 |

The validation and test subsets contain real images only.

This setup evaluates the classifier on held-out real images rather than evaluating it on synthetic images that were added to training.

### Classifier configuration

| Parameter | Value |
|---|---:|
| Architecture | ResNet-18 |
| Initial weights | ImageNet pretrained |
| Input resolution | 224 × 224 |
| Batch size | 32 |
| Maximum epochs | 15 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 4 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Random seed in reported runs | 42 |

### Evaluation metrics

The following metrics are used:

**Accuracy**

Measures the proportion of correctly classified test images.

**Macro Precision**

Calculates precision independently for each class and averages the four class scores.

**Macro Recall**

Calculates recall independently for each class and averages the four class scores.

**Macro F1-score**

Calculates the F1-score for each class and takes the unweighted average.

**Training Time**

Measures the time required to train the downstream classifier.

These metrics provide information about overall classification performance, class-wise performance, and computational cost.

---

## Experimental Results

The following values are reproduced from the project's committed comparison graphics.

### Classification performance

| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-score (%) | Training time (min) |
|---|---:|---:|---:|---:|---:|
| Baseline | 98.690 | 98.202 | 98.642 | 98.411 | 2.673 |
| LoRA Rank 8 | 98.589 | 98.735 | 98.082 | 98.379 | 2.611 |
| LoRA Rank 16 | 99.194 | 99.066 | 98.913 | 98.988 | 2.686 |

**Result interpretation**

- The Baseline configuration achieved 98.690% accuracy.
- LoRA Rank 8 achieved 98.589% accuracy.
- LoRA Rank 16 achieved the highest reported accuracy at 99.194%.
- Rank 16 also achieved the highest reported macro F1-score at 98.988%.
- The three reported training times are close, ranging from approximately 2.61 to 2.69 minutes.

These results suggest that the Rank 16 configuration performed best in the reported comparison. However, the difference should not be treated as statistically significant without repeated runs and an appropriate statistical analysis.

### Important reproducibility note

The values above are reported from the committed result graphics. The uploaded notebook outputs directly show the Baseline and Rank 8 classifier evaluations. The Rank 16 notebook confirms Rank 16 adapter training and synthetic generation, but the complete Rank 16 classifier evaluation is not independently established by the saved outputs reviewed here.

Before publishing, ensure that the Rank 16 classifier result is supported by its corresponding saved metrics file and that all three experiments use the same test set and evaluation procedure.

### Mean and standard deviation

The currently reported comparison contains one result per configuration. It does not provide multiple independent classifier runs.

Therefore, mean ± standard deviation is not reported yet.

For a more reliable evaluation, repeat each classifier experiment using multiple random seeds, for example:

- 42
- 123
- 2026

For each metric, calculate:

\[
\text{Mean} = \frac{1}{n}\sum_{i=1}^{n}x_i
\]

\[
\text{Standard Deviation} =
\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}
\]

Then report the results in the following format:

| Experiment | Accuracy (mean ± std) | Macro F1 (mean ± std) |
|---|---|---|
| Baseline | Pending repeated runs | Pending repeated runs |
| LoRA Rank 8 | Pending repeated runs | Pending repeated runs |
| LoRA Rank 16 | Pending repeated runs | Pending repeated runs |

Do not interpret a single seed as a mean or use it to calculate a standard deviation.

---

## Visualizations

The project includes visualizations for dataset inspection, synthetic image generation, and classifier performance.

### 1. Dataset samples

Displays representative real tomato leaf images from the four selected categories.

### 2. Class distribution

Shows the distribution of images across the selected classes.

### 3. Generated image samples

Displays synthetic images generated using:

- Baseline Stable Diffusion 1.5
- Stable Diffusion 1.5 + LoRA Rank 8
- Stable Diffusion 1.5 + LoRA Rank 16

Use the actual generated samples saved in the experiment output folders for the final comparison.

### 4. Training and validation curves

The downstream classifier records training and validation loss and accuracy across epochs.

These curves help inspect:

- Training convergence
- Validation performance
- Possible overfitting
- Early stopping behaviour

The Diffusers training process also records optimization-step loss values. A dedicated saved diffusion-loss plot should be added if the training logs are available.

### 5. Confusion matrices

Confusion matrices show how the classifier predicts each of the four tomato leaf categories and help identify class-wise misclassifications.

### 6. Performance comparison plots

The repository includes the following result graphics:

**Performance comparison**

![Performance comparison](results/performance_comparison.svg)

**Accuracy comparison**

![Accuracy comparison](results/accuracy_comparison.svg)

**Model metrics and training-time table**

![Model metrics table](results/model_metrics_table.svg)

**Note:** Ensure these SVG files are committed under the `results/` directory and that the filenames match the links above.

---

## Comparison and Ablation Study

### 1. Baseline versus LoRA fine-tuning

The first comparison investigates whether domain-specific LoRA fine-tuning changes the usefulness of generated images.

| Configuration | Model setup |
|---|---|
| Baseline | Pretrained Stable Diffusion 1.5 |
| LoRA Rank 8 | Stable Diffusion 1.5 with Rank 8 adapters |
| LoRA Rank 16 | Stable Diffusion 1.5 with Rank 16 adapters |

The same dataset categories and generation settings are used for the comparison.

### 2. LoRA rank ablation

The Rank 8 and Rank 16 experiments form a LoRA-rank ablation.

The purpose is to investigate how adapter rank affects:

- Adapter size
- Fine-tuning configuration
- Generated image appearance
- Downstream classification performance

The Rank 8 adapter is approximately 6.12 MB, whereas the Rank 16 adapter is approximately 12.20 MB.

This comparison studies the effect of increasing adapter capacity while retaining the same base diffusion model and principal training settings.

### 3. Real-only versus real-plus-synthetic ablation

A useful additional experiment is to compare training with real images only against training with real and synthetic images together.

| Experiment | Training data |
|---|---|
| Real-only control | 4,629 real images |
| Baseline augmentation | 4,629 real + 400 baseline synthetic images |
| Rank 8 augmentation | 4,629 real + 400 Rank 8 synthetic images |
| Rank 16 augmentation | 4,629 real + 400 Rank 16 synthetic images |

The real-only control is proposed as additional work and should not be considered completed unless its classifier experiment and metrics are added.

This experiment would help determine whether synthetic data improves downstream classification compared with using only real training images.

### 4. Comparison across datasets

The current experiments use the PlantVillage dataset and four tomato leaf categories.

A second dataset has not been evaluated in the reported experiments. Cross-dataset evaluation is a possible future extension.

### 5. Generative image-quality metrics

The current evaluation includes visual inspection and checks for invalid or missing generated images.

Additional quantitative generative metrics can strengthen the evaluation:

- **FID (Fréchet Inception Distance):** compares feature distributions of real and generated images. Lower values generally indicate closer distributions.
- **KID (Kernel Inception Distance):** measures distribution similarity using a kernel-based metric. Lower values generally indicate closer distributions.
- **LPIPS:** measures perceptual distance between meaningful image pairs. It requires a suitable pairing strategy and should not be interpreted as a standalone distribution-quality metric.

These metrics are suggested extensions and should only be listed as completed after they have actually been calculated.

---

## Reproducibility and Evaluation Notes

### Environment

The uploaded Rank 16 notebook records the following environment:

| Component | Recorded version or value |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| Diffusers | 0.42.0 |
| Transformers | 5.20.2 |
| Accelerate | 1.15.0 |
| PEFT | 0.21.2 |
| Datasets | 4.8.5 |
| GPU | NVIDIA Tesla T4 |
| GPU memory | Approximately 14.56 GB |

The runtime and package versions may differ across Kaggle sessions.

### Reproducibility considerations

For a reproducible experiment:

- Use the same PlantVillage source and selected class folders.
- Preserve the cleaned dataset manifest and split assignments.
- Use the same held-out real validation and test sets for every classifier experiment.
- Keep generation prompts and inference settings consistent.
- Record random seeds and training configurations.
- Save generated-image metadata and classifier metrics.
- Record the hardware and runtime used for training-time comparisons.
- Repeat classifier runs with multiple seeds before reporting mean ± standard deviation.

The Kaggle paths used in the notebooks are environment-specific and must be adjusted when running outside Kaggle.

---

## Repository Structure

The repository contains the experimental notebooks and result visualizations.

```text
Latent-Diffusion-LoRA/
│
├── README.md
│
├── GenAI-Diffusion vs Diffusion+lora(8).ipynb
│
├── GenAI-lora(16).ipynb
│
└── results/
    ├── performance_comparison.svg
    ├── accuracy_comparison.svg
    └── model_metrics_table.svg
```

The generated datasets, model checkpoints, and intermediate training files may be stored separately in the Kaggle working environment and are not necessarily included in this repository.

---

## Execution Instructions

### Requirements

The notebooks use Python and the following main libraries:

```bash
pip install torch torchvision
pip install diffusers transformers accelerate peft
pip install datasets safetensors
pip install numpy pandas matplotlib seaborn scikit-learn
pip install pillow tqdm
```

The exact compatible versions should be taken from the notebook environment.

### Running the experiments

1. Clone or download the repository.
2. Open the notebooks in Kaggle or a compatible Jupyter environment.
3. Enable a compatible GPU runtime.
4. Configure the PlantVillage dataset path.
5. Execute the dataset inspection and preprocessing cells.
6. Run the Baseline and LoRA Rank 8 experiment notebook.
7. Run the LoRA Rank 16 experiment notebook.
8. Verify the generated images and metadata.
9. Run the downstream classifier experiments.
10. Review the saved metrics, plots, and comparison outputs.

A Hugging Face access token may be useful for model downloads and higher download limits. Never commit access tokens or other credentials to the repository.

---

## Limitations and Future Work

The project provides an experimental comparison of pretrained Stable Diffusion and LoRA fine-tuning for domain-specific synthetic image generation.

The following extensions can improve the strength of the evaluation:

1. **Multiple independent runs:** Repeat classifier experiments using multiple seeds and report mean ± standard deviation.
2. **Real-only baseline:** Compare real-only classifier training against real-plus-synthetic training.
3. **Generative quality metrics:** Calculate FID and KID using consistent real reference images and generated-image counts.
4. **Diffusion loss visualization:** Save and plot the diffusion training loss against optimization steps.
5. **Expanded dataset evaluation:** Test the approach on an additional image dataset.
6. **Class-wise analysis:** Compare confusion matrices and per-class precision, recall, and F1-score.
7. **More adapter configurations:** Investigate additional LoRA ranks while keeping other training settings consistent.

---

## Conclusion

This project investigates domain-specific synthetic image generation using Stable Diffusion 1.5 and Low-Rank Adaptation.

By comparing a pretrained baseline with LoRA Rank 8 and Rank 16 configurations, the experiments explore how parameter-efficient fine-tuning can adapt a general-purpose diffusion model to tomato leaf disease images.

The generated images are also incorporated into a downstream ResNet-18 classification workflow to study their potential usefulness for image classification.

The reported comparison shows that the Rank 16 configuration achieved the highest classification accuracy and macro F1-score among the three listed configurations. Further repeated experiments, a real-only control, and quantitative image-generation metrics would provide stronger evidence about the consistency and contribution of synthetic data.

---

## License and Dataset Usage

The repository and dataset licenses should be checked separately before redistribution or commercial use.

The PlantVillage dataset, pretrained Stable Diffusion model, generated images, and project code may have different licensing and usage conditions. No common license should be assumed for all of these components.
