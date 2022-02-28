This repository is supplementary material to
# OAB - An Open Anomaly Benchmark Framework for Unsupervised and Semisupervised Anomaly Detection on Image and Tabular Data Sets


```
.
├── OAB_SupplementaryMaterial.pdf
├── README.md
├── notebooks
│   ├── benchmark_image
│   ├── benchmark_tabular
│   ├── comparing_algorithms
│   ├── metrics
│   ├── own_algorithm_semisupervised
│   ├── own_algorithm_unsupervised
│   ├── replicate_campos
│   └── reproducibility
├── oab
└── test
```

Supplementary material for the paper with additional information about the data sets supported by `oab` and experiment results with more evaluation metrics can be found at `OAB_SupplementaryMaterial.pdf`.

`oab` contains the code for the oab package, and `test` contains `pytest` tests.

`notebooks` provides Jupyter notebooks with example use cases for `oab_v0.1`:
- `benchmark_image` contains the notebooks that were used to run the image experiments. This folder also contains the code for the algorithms used in these experiments.
- `benchmark_tabular` contains the notebooks that were used to run the tabular data experiments. This folder also contains the code for AE+LOF.
- `comparing_algorithms` contains a notebooks that explains how algorithms' performances can be compared with `oab`.
- `metrics` contains a notebook showcasing how different metrics are selected.
- `own_algorithm_semisupervised` provides an example notebook of how an own semisupervised anomaly detection algorithm can be evaluated with `oab`.
- `own_algorithm_unsupervised` provides an example notebook of how an own unsupervised anomaly detection algorithm can be evaluated with `oab`.
- `replicate_campos` contains a notebooks that was used to replicate results from the Campos et al. paper.
- `reproducibility` contains notebooks showcasing how `oab` can be used to replicate a sampling and preprocessing procedure, making it easy to reproduce results and allowing custom preprocessing, sampling strategies, etc.

To run the notebooks locally in a virtual environment, install the dependencies from `requirements.txt` and run the notebooks in jupyter lab. (In case there are problems with executing within the virtual environment in jupyter lab, [this](https://janakiev.com/blog/jupyter-virtual-envs/) might help.)
Alternatively, the notebooks can also be run on [Colab](https://drive.google.com/drive/folders/1ZKEHmldEsLhK6fhhhNgr_YthLUX_xusC?usp=sharing).

`oab_v0.3` improved `oab_v0.2` interms of storing benchmarking recipes . In `oab_v0.2` a different recipe file(.yaml) was created for every algorithm and dataset used in the benchmark, containing their corresponding information,  for reproducing the benchmark run. In contrast, `oab_v0.3` provides a functionality of creating only 1 recipe file per benchmark run storing information of all algorithms and datasets used in that run.


There is a  jupyter notebook for the cases of Semisupervised and Unsupervised Anomaly detection, located  at `/notebooks/benchmark_image` and `/notebooks/benchmark_tabular` in case of image datasets and tabular datasets respectively, which  provide structured explaination of every step for benchmarking in just one jupyter notebook.
