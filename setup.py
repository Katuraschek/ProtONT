import setuptools 
  
with open("README.md", "r") as fh: 
    description = fh.read() 
  
setuptools.setup( 
    name="ProtONT", 
    version="0.0.1", 
    author="Katharina Juraschek", 
    author_email="Katharina.Juraschek@childrens.harvard.edu", 
    packages=["ProtONT"], 
    description="LOESS curve and volcano plot generator with protein LFQ ontogeny data.", 
    long_description=description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/gituser/empty", 
    license='BCH', 
    python_requires='>=3.8', 
    install_requires=["pandas", "numpy", "matplotlib", "scipy", "itertools", "statsmodels", "os", "sklearn"]
)