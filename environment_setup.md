# Environment Setup Options

You have two ways to set up your environment:

---

## Option 1: Create the Environment Manually

```bash
# Create environment with specific Python version
conda create --name py386 python=3.8.6 -c conda-forge
conda activate py386

# Install required packages
conda install pandas
conda install matplotlib
conda install tqdm
pip install wfdb

# Fix dependency issues
pip check   # Will report missing pyyaml
pip install pyyaml
pip check   # Should show no broken requirements

# Additional packages
conda install scikit-learn
conda install -c conda-forge scikit-image
pip install fastai==1.0.61
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install requests==2.25.1 charset-normalizer==2.0.12 --force-reinstall
```

---

## Option 2: Create the Environment from Files

```bash
# Create environment directly from conda spec
conda create --name py386_from_file --file conda-spec.txt
conda activate py386_from_file

# Install pip dependencies from requirements file
pip install -r pip-requirements.txt
```
