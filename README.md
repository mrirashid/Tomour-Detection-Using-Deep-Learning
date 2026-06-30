# Brain Tumor Detection Using Deep Learning

![Designer (3)](https://github.com/user-attachments/assets/06d131d7-8d30-4c42-8198-7654ce8aebc2)

## Table of Contents
- Description
- Inspiration
- Source of Dataset
- Interesting Facts from EDA
- Technical Approach
- Get Started
  - Pre-installation
  -  Setup
- Report
- Results
- Challenges
- Contributors

## Description
This project applies deep learning techniques to detect brain tumors from MRI images. Using two advanced models, EfficientNetB0 and EfficientNetB3, this project achieved impressive accuracy in classifying tumors into four categories: glioma tumor, no tumor, meningioma tumor, and pituitary tumor .

Paper link: https://ieeexplore.ieee.org/abstract/document/11276886

## Inspiration
This experiment is part of a research endeavor to explore deep learning applications in medical imaging. The insights gathered from this project will be used in writing a research paper on the topic.

## Source of Dataset
[Dataset Link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
## Interesting Facts from Exploratory Data Analysis
- Glioma tumors are the most frequently occurring in the dataset.
- Image intensity and contrast differences were noticeable across tumor types.
- Data augmentation helped balance the dataset and reduce overfitting.

## Technical Approach
Two EfficientNet models were used:
- EfficientNetB0: A smaller model with lower computational cost but highly efficient for image classification.
- EfficientNetB3: A more complex model with greater capacity for high-dimensional feature learning.


###  Get Started
- Pre-installation
Ensure that you have the following libraries installed:
```bash 
!pip install mplcyberpunk
!pip install tensorflow keras


```

- Set-up
Clone this repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

```
## Report

[Report.pdf](https://github.com/user-attachments/files/17302300/Report.pdf)
## Results
Both models achieved 97% accuracy, making them highly reliable for brain tumor detection. The confusion matrices and classification reports indicate precise and robust model performance across all tumor classes.


## Challenges
One of the main challenges was choosing the right model architecture. Several models, including ResNet and VGG, were tested before settling on EfficientNetB3 and EfficientNetB0. Data preprocessing and augmentation techniques also played a crucial role in enhancing model performance.


## Contributing
- Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Developed by [MD Rashidul Islam](https://github.com/mrirashid/)

