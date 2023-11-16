# ConeQuest
ConeQuest is the first expert-annotated publicly available dataset for cone segmentation across three different regions on Mars, along with metadata for each sample.


![bm_1](https://github.com/kerner-lab/ConeQuest/assets/46327378/e6c37a7e-ae1c-4a1f-956f-cc25ea984f30)

- ConeQuest can be accessed from [Zenodo](https://zenodo.org/records/10077410).

- The ConeQuest dataset has a total of 13,686 patches from 8 different subtiles across 3 regions.

- Metadata of each CTX tile across three regions used in the creation of ConeQuest:

<img width="1028" alt="Screenshot 2023-11-11 at 6 26 39 PM" src="https://github.com/kerner-lab/ConeQuest/assets/46327378/dd1e2f27-39e8-4e0f-b4ad-40b6be3e8338">


### Getting Started

#### Environment Setup

```bash
conda env create -f conequest_env.yml
```

#### Download ConeQuest from Zenodo

```bash
./download_data.sh
```


### License
ConeQuest has a [Creative Commons Zero v1.0 Universal](https://github.com/kerner-lab/ConeQuest/blob/main/LICENSE) license.

### Citation

If you use ConeQuest in your research, please use the following citation:
```
@inproceedings{
    purohit2024conequest,
    title={ConeQuest: A Benchmark for Cone Segmentation on Mars},
    author={Mirali Purohit and Jacob Adler and Hannah Kerner},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2024}
}
```
