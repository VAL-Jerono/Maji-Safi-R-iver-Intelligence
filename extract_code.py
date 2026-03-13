import json

def extract_code(ipynb_path, py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open(py_path, 'w', encoding='utf-8') as f:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                f.write(f"# CELL {i}\n")
                f.write(''.join(cell.get('source', [])))
                f.write('\n\n')

extract_code('Jupyter Notebook Package/Landsat_Demonstration_Notebook.ipynb', 'Jupyter Notebook Package/Landsat_Demonstration.py')
extract_code('Jupyter Notebook Package/TerraClimate_Demonstration_Notebook.ipynb', 'Jupyter Notebook Package/TerraClimate_Demonstration.py')
extract_code('Jupyter Notebook Package/Benchmark_Model_Notebook.ipynb', 'Jupyter Notebook Package/Benchmark_Model.py')
