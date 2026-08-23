from pathlib import Path
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.cells=[nbf.v4.new_markdown_cell("# Predicting Smartphone Addiction — EDA\n\nExecuted, reproducible analysis. Outputs are saved under `reports/`."),nbf.v4.new_code_cell("%cd ..\n%run notebooks/eda_build.py")]
nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb,Path("notebooks/01_eda.ipynb"))
