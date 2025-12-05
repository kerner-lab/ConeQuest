# ConeQuest [WACV 2024 Oral]
ConeQuest is the first expert-annotated publicly available dataset for cone segmentation across three different regions on Mars, along with metadata for each sample.


![bm_1](https://github.com/kerner-lab/ConeQuest/assets/46327378/e6c37a7e-ae1c-4a1f-956f-cc25ea984f30)

- ConeQuest can be accessed from [Zenodo](https://zenodo.org/records/10077410).

- The ConeQuest dataset has a total of 13,686 patches from 8 different subtiles across 3 regions.

- Metadata of each CTX tile across three regions used in the creation of ConeQuest:

<img width="1028" alt="Screenshot 2023-11-11 at 6 26 39 PM" src="https://github.com/kerner-lab/ConeQuest/assets/46327378/dd1e2f27-39e8-4e0f-b4ad-40b6be3e8338">

More details about ConeQuest are available in [this paper](https://arxiv.org/abs/2311.08657).

##

### Getting Started

#### Environment Setup

```bash
conda env create -f conequest_env.yml
```

#### Download ConeQuest from Zenodo

```bash
./download_data.sh
```

#### Training and Testing models
Utilize the arguments described in _main.py_ to train and test models for various configurations. A few important arguments are explained below:
- Provide _training_type_ argument 1 or 2 to train a model for benchmarks 1 and 2, respectively.
- In the _training_data_list_, provide a list (string separated with a comma) of region/s or size/s on which model will be trained, e.g., "Isidis Planitia, Hypanis" or "small, medium".
- Use _if_training_positives_ to train your model only on positive samples.

```bash
python main.py \
    --if_training \
    --training_type 1 \
    --training_data_list "Isidis Planitia" \
    --train_model DeepLab
```

- In the _eval_data_, provide which region or size to evaluate.
```bash
python main.py \
    --if_testing \
    --training_type 1 \
    --training_data_list "Isidis Planitia, Hypanis" \
    --train_model DeepLab
    --eval_data Hypanis
```

### License
ConeQuest has a [Creative Commons Zero v1.0 Universal](https://github.com/kerner-lab/ConeQuest/blob/main/LICENSE) license.

### Citation

If you use ConeQuest in your research, please use the following citation:

```
@InProceedings{Purohit_2024_WACV,
    author={Purohit, Mirali and Adler, Jacob and Kerner, Hannah},
    title={ConeQuest: A Benchmark for Cone Segmentation on Mars},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month={January},
    year={2024},
    pages={6026-6035}
}
```
### Contact Information
Please reach out to Mirali Purohit [mpurohi3@asu.edu](mpurohi3@asu.edu), if you have any queries or issues regarding ConeQuest.

