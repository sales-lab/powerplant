# PowerPlant

PowerPlant is a Python package that leverages deep learning to forecast
the success of DNA extraction from herbarium samples. This tool is
designed to assist botanical researchers in optimizing their selection
of herbarium specimens for genomic studies.

## Overview

PowerPlant employs a deep learning algorithm that integrates multiple
data sources to predict ancient DNA extraction success:

1.  Morphological features from scanned herbarium images
2.  Sample color information
3.  Metadata including sample age and locality
4.  DNA quantity metrics from previously processed samples

Trained on a dataset of approximately 2,000 herbarium specimens from the
PAFTOL project, spanning nearly two centuries (1832 to present),
PowerPlant aims to revolutionize the approach to working with
herbarium-derived DNA.

## Requirements

- Linux or macOS operating system.
- Python 3.11 (later versions may not be fully supported by some
  dependencies).
- GPU support recommended for optimal performance.

## Installation

PowerPlant integrates several deep learning tools. While most are
distributed as Python packages and can be installed via `pip` or
`conda`, the image segmentation component relies on the
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) framework, which
requires specific installation steps.

1.  Clone the PowerPlant repository:

``` sh
git clone https://github.com/sales-lab/powerplant.git
```

2.  Create and activate a virtual environment:

``` sh
cd powerplant
python3 -m venv .venv
source .venv/bin/activate
```

3.  Install PowerPlant:

``` sh
pip install ./
```

4.  Install PaddleSeg:

``` sh
cd vendor
sh install-paddleseg.sh
```

**Note**: The default installation is CPU-only. For GPU support, modify
the script to install the appropriate
[PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)
and
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/docs/install.md#22-install-paddleseg)
variants for your hardware.

## Usage

### Image Segmentation

PowerPlant processes herbarium sheet images in JPEG format (with `.jpg`
extension).

The package automatically performs two key operations on your images: -
Segmentation to isolate plant material and remove extraneous elements
such as annotations, labels, stamps, and envelopes. - Resizing of images
so that the longest side is at most 1024 pixels long.

Copy your original herbarium sheet images to the `images/original`
directory, then run the following command:

``` sh
powerplant-segment
```

The processed images (segmented and resized) will be stored in the
`images/masked` directory.

### Prediction of DNA Yield

PowerPlant employs a convolutional neural network (CNN) coupled with
metadata analysis to predict DNA yield from herbarium specimens. This
dual-input model processes both segmented images and associated specimen
data to generate accurate yield estimates.

To use this feature:

1.  Ensure your segmented herbarium images are stored in the
    `images/masked` directory. These should be the output from the
    preprocessing step described in the [Image
    Segmentation](#image-segmentation) section.
2.  Prepare your metadata in a CSV file named `samples.csv` and place it
    in the `metadata` directory. This file should contain relevant
    information for each specimen, including:
    - Specimen age;
    - Location of sample collection;
    - Taxonomic information.

An example `samples.csv` file is included in this repository to guide
you in formatting your metadata correctly.

To train the prediction model, run the following command:

``` sh
powerplant-train
```

This script processes the images and metadata from the dataset
directory, trains the machine learning model, and saves the trained
model in the `checkpoints/prediction` directory.

## License

GNU Affero General Public License, version 3.

## Contact

For questions and support, please [open an
issue](https://github.com/sales-lab/powerplant/issues) on our GitHub
repository.
