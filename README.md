Installation using Conda
You need to have Python 3.9 or newer installed on your system.

Create a conda environment and activate it.

conda create -n ProtONT
conda activate ProtONT
With git installed (recommended):
Clone this repository using git clone https://github.com/Katuraschek/ProtONT.git and navigate there to install it in the newly created conda environment (use -e for editable):

pip -e install .
After updates on the github, you can use the newest version of the code by running git pull in the cloned directory.

Without git installed:
Directly install from github using

pip install git+https://github.com/Katuraschek/ProtONT.git#egg=ProtONT
You can verify that devipep was installed by using conda list. To incorporate the newest changes from github you can run

pip install -U --force-reinstall git+https://github.com/Katuraschek/ProtONT.git#egg=ProtONT
This is less convenient as you have to reinstall the entire package from the conda prompt each time.

Then you can use the package from your conda environment, e.g. to run the example notebook demonotebook.ipynb. In order to run jupyter notebooks, you can install jupyter in your environment using

conda install jupyter
Contact
Katharina.Juraschek@childrens.harvard.edu.

Please use the Issues section for bug reports and feature requests.
