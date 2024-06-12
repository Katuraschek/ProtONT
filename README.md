# Installation using Conda

To install ProtONT, you need to have Python 3.9 or newer installed on your system.

## 1. Create and Activate a Conda Environment

Create a Conda environment and activate it:

```bash
conda create -n ProtONT
conda activate ProtONT
```

## 2. Install with Git (Recommended)

Clone this repository with git clone https://github.com/Katuraschek/ProtONT.git and navigate there to install it in the newly created Conda environment (use -e for editable):

```bash
pip install -e .
```

After updates on GitHub, you can use the newest version of the code by running git pull in the cloned directory.

## 3. Install without Git

Install directly from GitHub with:

```bash
pip install git+https://github.com/Katuraschek/ProtONT.git#egg=ProtONT
```

You can verify that ProtONT was installed by using conda list. To incorporate the newest changes from GitHub, you can run:

```bash
pip install -U --force-reinstall git+https://github.com/Katuraschek/ProtONT.git#egg=ProtONT
```

This is less convenient as you have to reinstall the entire package from the Conda prompt each time.

## 4. Usage

You can use the package from your Conda environment, e.g., to run the example notebook demonotebook.ipynb. To run Jupyter Notebooks, you can install Jupyter in your environment using:

```bash
conda install jupyter
```

## Contact

For questions and suggestions, please contact Katharina.Juraschek@childrens.harvard.edu.

Please use the "Issues" section for bug reports and feature requests.
