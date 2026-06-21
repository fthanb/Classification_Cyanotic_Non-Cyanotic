# Classification of CHD Cyanotic and Non-Cyanotic using CNN+LSTM

##  Project Overview

This project aims to leverage deep learning models for the classification of Congenital Heart Disease (CHD) using 3D Computed Tomography (CT) scan data. The primary objective is to accurately categorize patient scans into distinct classes: Cyanotic, Non-Cyanotic, and Normal(non-CHD). The research explores deep learning architectures to understand their effectiveness in this medical imaging task. This includes an **Convolutional Neural Network (CNN)** for 2D image slice classification, as well as **hybrid CNN-Long Short-Term Memory (LSTM) models** designed to process the inherent sequential information within 3D CT scan volumes. **K-Fold Cross-Validation** strategy is also implemented

## Tools & Requirements

* **Python 3.11**
* **PyTorch 2.5.1** 
* **NumPy** 
* **Pillow (PIL)** 
* **Nibabel** 
* **Scikit-learn**
* **Matplotlib & Seaborn** 
* **tqdm** 

## Data Source
The dataset used in this project is from Kaggle:
[https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd).

To extract the dataset files:
1. Download the zipped file from the Kaggle link.
2. You might find the downloaded file has a non-standard extension (e.g., ".change2zip"). **Rename this file's extension to ".zip"** (e.g., `ImageCHD_dataset.change2zip` becomes `ImageCHD_dataset.zip`).
3. **Extract all files** from the renamed `.zip` archive.

## Normalizing Data (Pre-process)
* The original dataset contains 110 patients with various types of CHD, as detailed in the `imageCHD_dataset_info.xlsx` file.
* For this project, data from 72 specific patients was selected based on categories defined in a CHD paper: "Sianotik," "Non-Sianotik," and "Normal".
    * **Sianotik:** `[1008, 1010, 1012, 1015, 1028, 1037, 1046, 1050, 1064, 1074, 1085, 1092, 1099, 1105, 1111, 1113, 1120, 1125, 1129, 1141, 1145, 1146, 1147, 1150, 1158, 1178]`
    * **Non-Sianotik:** `[1001, 1002, 1007, 1011, 1014, 1018, 1019, 1020, 1025, 1029, 1033, 1035, 1036, 1041, 1047, 1061, 1070, 1079, 1103, 1109, 1132, 1133, 1135, 1139, 1140, 1148]`
    * **Normal:** `[1003, 1005, 1032, 1051, 1062, 1063, 1066, 1067, 1072, 1078, 1080, 1083, 1101, 1116, 1117, 1119, 1127, 1128, 1143, 1144]`

* The number of frames (slices) for each patient varies within the raw NIfTI files. To ensure consistency for the deep learning models, all patient scans are standardized to a uniform length of 275 frames.

## CNN Basic (`cnn.ipynb`)

First try is a basic Convolutional Neural Network for classifying individual CT scan slices

* **Data Handling:**
    * The dataset path (`root_dir`) is set to `C:\Users\risuser\Documents\RISET_FATHAN\dataset`, which is expected to contain `ct_<patient_id>` folders with the preprocessed 275 PNG image slices.
    * **Patient-level splitting:** Patients are first divided into train 80%, validation 10%, and test sets 10% based on specific counts (Normal: 16/2/2; Sianotik: 20/3/3; Non-Sianotik: 20/3/3). This ensures that all slices from a given patient are contained within a single split (e.g., if patient 1003 is in the training set, all 275 of their slices will only be used for training).
    * A custom `CTScanDataset` loads individual PNG images, associating each with its patient's pre-assigned label (Sianotik: 0, Non-Sianotik: 1, Normal: 2).
    * **Image Augmentation:** A `transforms.Compose` pipeline is applied to the training images, including `transforms.Resize((128, 128))`, `transforms.RandomHorizontalFlip(p=0.5)`, `transforms.RandomRotation(degrees=15)`, and `transforms.ColorJitter(brightness=0.1, contrast=0.1)`. These augmentations introduce variability to the training data, helping the model generalize better.
    * **Normalization:** Images are converted to tensors and normalized to a range of [-1, 1] using `transforms.Normalize((0.5,), (0.5,))`.
    * `DataLoader`s are set up with a `batch_size` of 32 for training, validation, and testing.

* **Model Architecture (`EnhancedCNN`):**
    * **Input Layer:** Accepts individual grayscale PNG image slices of size 128x128 pixels (after resizing transform). The input shape to the model is typically $(Batch\_size, 1, 128, 128)$.
    * **Convolutional Blocks (`self.conv_layers`):** The model uses three sequential blocks for feature extraction.
        * **Block 1:** `nn.Conv2d(1, 16, kernel_size=3, padding=1)`, followed by `nn.BatchNorm2d(16)`, `nn.ReLU()`, `nn.MaxPool2d(2, 2)`, and `nn.Dropout(0.3)`.
        * **Block 2:** `nn.Conv2d(16, 32, kernel_size=3, padding=1)`, followed by `nn.BatchNorm2d(32)`, `nn.ReLU()`, `nn.MaxPool2d(2, 2)`, and `nn.Dropout(0.4)`.
        * **Block 3:** `nn.Conv2d(32, 64, kernel_size=3, padding=1)`, followed by `nn.BatchNorm2d(64)`, `nn.ReLU()`, `nn.MaxPool2d(2, 2)`, and `nn.Dropout(0.5)`.
    * **Flattening:** The 3D output of the last convolutional/pooling layer (which is $64 \times 16 \times 16$) is flattened into a 1D vector.
    * **Fully Connected Layers (`self.fc_layers`):** These layers perform the final classification.
        * `nn.Linear(64 * 16 * 16, 256)`
        * `nn.Dropout(0.6)`
        * `nn.ReLU()`
        * `nn.Linear(256, 128)`
        * `nn.Dropout(0.5)`
        * `nn.ReLU()`
        * `nn.Linear(128, 3)` (for 3 output classes).
    * **Forward Pass (`forward` method):** The input `x` is passed sequentially through `self.conv_layers`, then flattened, and finally passed through `self.fc_layers`.

* **Training and Evaluation:**
    * **Loss Function:** `nn.CrossEntropyLoss` is used, with `weight` calculated based on inverse class frequencies in the training set to address class imbalance.
    * **Optimizer:** `torch.optim.AdamW` is used with a learning rate of `0.00005` and `weight_decay` of `1e-4`.
    * **Scheduler:** `optim.lr_scheduler.ReduceLROnPlateau` monitors the validation loss (`mode='min'`) and reduces the learning rate by a `factor=0.3` if no improvement is seen for `patience=3` epochs.
    * **Epochs:** The model is trained for 50 epochs.
    * **Model Saving:** The model's state dictionary is saved as `best_model.pth` whenever a new `best_val_loss` is achieved during training.
    * **Performance Tracking:** Training and validation loss and accuracy are recorded per epoch.
    * **Evaluation:** and **Test Accuracy:** are showned in the code.

## CNN+LSTM with K-fold Cross Validation (`cnnlstm_kfold.ipynb`)

This section details the CNN-LSTM model implemented with K-Fold Cross-Validation, more robust technique for model evaluation.

* **Data Handling:**
    * The `dataset_dir` is set to `D:\coding\SKRIPSI\CLASSIFICATION MODEL\dataset`, which is expected to contain `ct_<patient_id>` folders with the preprocessed 275 PNG frames.
    * **Patient-level splitting with Stratified K-Fold:** The project uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` to divide the 72 patient IDs into 5 folds. This ensures that the proportion of each class (Sianotik, Non-Sianotik, Normal) is maintained across all training and validation folds.
    * A `CT3DDataset` (identical to the one in `cnnlstm.ipynb`) loads all 275 frames for each patient as a volume. Each frame is converted to a tensor (implicitly normalizing to [0,1]).
    * `DataLoader`s are set up with a `batch_size` of 1 for both training and validation within each fold.

* **Model Architecture (`CNN_LSTM`):**
    * The `CNN_LSTM` class used here is **structurally almost identical** to the one in `cnnlstm.ipynb`.
    * **Key Addition:** It explicitly includes a `nn.Dropout(0.5)` layer applied *before the LSTM input*. This is a regularization technique designed to prevent overfitting by randomly setting a fraction of input units to zero during training.
    * Other components (CNN layers, LSTM configuration, `feature_dim`, `fc` layer) remain the same.

* **Training and Evaluation:**
    * **K-Fold Loop:** The entire training and validation process is wrapped in a loop that iterates through each of the 5 folds. In each fold:
        * A new instance of the `CNN_LSTM` model is initialized.
        * **Loss Function:** `nn.CrossEntropyLoss()`.
        * **Optimizer:** `torch.optim.Adam` with a learning rate of `1e-4`.
        * **Epochs:** Each fold is trained for 20 epochs.
        * **Model Saving:** The model's state dictionary is saved for each fold (e.g., `cnn_lstm_fold1.pth`, `cnn_lstm_fold2.pth`, etc.).
        * **Performance Tracking:** Training and validation loss and accuracy are recorded per epoch for each fold and plotted.
        * **Evaluation (per fold):** After training for a fold, the model is evaluated on that fold's validation set (`val_loader`). A confusion matrix is generated and displayed for each fold's validation performance.

* **Analysis of Results:**
    The results for the CNN-LSTM K-Fold Cross-Validation model are presented for each of the 5 folds directly within the notebook's output. **You can see the results shown in the code**, including printed epoch-wise training and validation accuracies and losses, as well as plotted loss/accuracy curves and confusion matrices for each fold.

    Upon reviewing the results:
    * **Training Accuracy:** For most folds, the training accuracy quickly reaches 1.0000 (100%), often within the first few epochs. This indicates that the model is very effective at learning the patterns in the training data, but also raises a flag for potential overfitting given the small batch size (1 patient per batch) and relatively small dataset.
    * **Validation Accuracy:** The validation accuracy is significantly lower and shows considerable variability across the folds:
        * Fold 1: 0.4000
        * Fold 2: 0.4000
        * Fold 3: 0.4286
        * Fold 4: 0.3571
        * Fold 5: 0.7143
  * This variability, particularly the much higher accuracy in Fold 5, suggests that the model's performance is sensitive to the specific patients included in the validation set. The relatively low average validation accuracy (around 40-50%) indicates that the model might struggle to generalize well to unseen patient data, especially given the rapid 100% training accuracy. The confusion matrices for each fold provide a detailed breakdown of correct and incorrect classifications per class. 
