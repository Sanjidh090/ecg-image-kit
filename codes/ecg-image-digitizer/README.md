# `ecg-image-digitizer`
***Software tools for ECG image processing and digitization***

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Python Functions

### Core Functions

| File | Description |
|---|---|
[ecg_grid_size_from_paper.py](ecg_grid_size_from_paper.py)| ECG grid size estimate from paper size and image width |
[ecg_gridest_margdist.py](ecg_gridest_margdist.py)| ECG grid size estimation - marginal image intensity approach |
[ecg_gridest_spectral.py](ecg_gridest_spectral.py)| ECG grid size estimation - spectral approach |
[ecg_gridest_matchedfilt.py](ecg_gridest_matchedfilt.py)| ECG grid size estimation - matched filter-based approach |
[image_to_sequence.py](image_to_sequence.py)| Extract time-series from an ECG segment |
[tanh_sat.py](tanh_sat.py)| Saturate signal/image intensity with a tanh function |

### Test Scripts

| File | Description |
|---|---|
[test_ecg_grid_size_estimator.py](test_ecg_grid_size_estimator.py)| A test script for running and comparing the grid size estimation methods |
[test_ecg_sequence_extraction.py](test_ecg_sequence_extraction.py)| A test script for time-series extraction from an image segment |

### Additional Tools

| Directory | Description |
|---|---|
[ROI](roi)| ECG lead detection with YOLOv7 |

## Usage Examples

### Example 1: Estimate grid size from paper dimensions

```python
import numpy as np
from PIL import Image
from ecg_grid_size_from_paper import ecg_grid_size_from_paper

# Load an ECG image
img = np.array(Image.open('path/to/ecg_image.jpg'))

# Define paper size (11 x 8.5 inches)
coarse_grid, fine_grid = ecg_grid_size_from_paper(img, 11.0, 'in')

print(f"Coarse grid size: {coarse_grid:.2f} pixels")
print(f"Fine grid size: {fine_grid:.2f} pixels")
```

### Example 2: Estimate grid size using spectral method

```python
from ecg_gridest_spectral import ecg_gridest_spectral

# Estimate grid size with default parameters
grid_sizes_hor, grid_sizes_ver = ecg_gridest_spectral(img)

print(f"Horizontal grid sizes: {grid_sizes_hor}")
print(f"Vertical grid sizes: {grid_sizes_ver}")
```

### Example 3: Extract time-series from ECG image segment

```python
from image_to_sequence import image_to_sequence

# Extract sequence using moving average method
data = image_to_sequence(img, 'dark-foreground', 'moving_average', 
                        windowlen=5, plot_result=True)
```

### Example 4: Run complete grid estimation test

```python
# Run the test script
python test_ecg_grid_size_estimator.py
```

## Citation
Please include references to the following articles in any publications:

1. Kshama Kodthalu Shivashankara, Deepanshi, Afagh Mehri Shervedani, Matthew A. Reyna, Gari D. Clifford, Reza Sameni (2024). ECG-image-kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. In Physiological Measurement. IOP Publishing. doi: [10.1088/1361-6579/ad4954](https://doi.org/10.1088/1361-6579/ad4954)


2. ECG-Image-Kit: A Toolkit for Synthesis, Analysis, and Digitization of Electrocardiogram Images, (2024). URL: [https://github.com/alphanumericslab/ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit)

## Contributors
- Deepanshi, Department of Biomedical Informatics, Emory University, GA, US
- Kshama Kodthalu Shivashankara, School of Electrical and Computer Engineering, Georgia Institute of Technology, Atlanta, GA, US
- Matthew A Reyna, Department of Biomedical Informatics, Emory University, GA, US
- Gari D Clifford, Department of Biomedical Informatics and Biomedical Engineering, Emory University and Georgia Tech, GA, US
- Reza Sameni (contact person), Department of Biomedical Informatics and Biomedical Engineering, Emory University and Georgia Tech, GA, US

## Contact
Please direct any inquiries, bug reports or requests for joining the team to: [ecg-image-kit@dbmi.emory.edu](ecg-image-kit@dbmi.emory.edu).


![Static Badge](https://img.shields.io/badge/ecg_image-kit-blue)




