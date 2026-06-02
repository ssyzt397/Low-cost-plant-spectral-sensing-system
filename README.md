# Low-cost-plant-spectral-sensing-system
A low-cost, non-invasive plant monitoring system developed as a Bachelor of Engineering Final Year Project.

This project combines multispectral sensing, environmental monitoring, and machine learning to classify leaf health conditions based on spectral characteristics.

# Repository Structure
├── Arduino/
│   └── Sensor acquisition programs
│
├── model python code/
│   └── code for different models
│
├── models/
│   └── Trained learning models for use in reality life
│
├── Test result/
│   └── Validation results
│
├── spectral.xlsx
│   └── Training dataset
│
├── Test spectral.xlsx
│   └── Validation dataset
│
├── desktop_app.py
│   └── Desktop application source code
│
├── desktop_app.spec
│   └── PyInstaller configuration
│
├── Confusion_matrix.py
│   └── Confusion matrix generation
│
└── picture.py
    └── leaf state generation figure
    
# Download and Run
Ready-to-Use Application
Download:
Leaf Monitoring App v1.0 from the Releases section.
Steps:
Download dist.zip
Extract the archive
Run:
leaf_monitoring.exe 
No Python installation required.

# Desktop Application
input:
Single Sample Prediction
Users can manually enter spectral values:
415 nm
445 nm
480 nm
515 nm
555 nm
590 nm
630 nm
680 nm
Batch Prediction
Users can upload Excel files for batch analysis.

Outputs include:
Predicted labels
Recommended actions
Exportable results
Recommended Actions
