requirements.txt	Python 3.12 virtual environment dependency file
co-occa.ipynb		Jupyter Notebook for extracting co-occurrence matrices from FoodSeg103
occurrence-food103.npy	Precomputed co-occurrence matrix for FoodSeg103 dataset
occurrence-recipe1M.npy	Precomputed co-occurrence matrix for Recipe1M+ dataset
auto.py			Main training and validation script


Data Preparation：
Unzip foodseg103-256.zip and copy its contents to /tmp/.


Execution Instructions：

bash
python auto.py --encoder tu-caformer_m36 --coocu False
With FoodSeg103 co-occurrence matrix:

bash
python auto.py --encoder tu-caformer_m36 --coocu True --decay 100
