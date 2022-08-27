# ML4PDR

```bash
├── README.md
├── code
│   ├── aiger_tools # Tools for convert .aig
│   ├── bmc.py # Implementation of BMC (used in PDR)
│   ├── config.py # Input parameters
│   ├── data_gen.py # Generate graph from aiger
│   ├── generate_aag.py # convert .aig to .aag
│   ├── main.py # Main function to run PDR in certain aiger files
│   ├── mlp.py # Simple implementation of MLP
│   ├── model.py # Parser for parse .aag into z3
│   ├── neuro_predessor.py # Neural Network model to train and inference
│   ├── pdr.py # Implementation of PDR
│   ├── run_slurm.py # This file could be used when run the program on slurm
│   ├── solver.py # Z3, Minisat (used by PDR)
│   ├── ternary_sim.py # Implemtation of ternary simulation
│   └── train.py # Training file for data processing
├── dataset # Dataset of training and validation
│   ├── Cora
│   ├── Cora2
│   ├── ILAng_pipeline
│   ├── aag4train
│   ├── edgelist
│   ├── eval
│   ├── fuzzing_aig
│   ├── generalization
│   ├── generalization_old
│   ├── generalize_pre
│   ├── hwmcc07_amba
│   ├── hwmcc07_tip
│   ├── tmp
│   ├── toy_experiment
│   └── train
├── deps # Some dependent modules
│   └── PyMiniSolvers
├── log # Contains logs file
│   ├── RL_spec3-and-env.txt
│   ├── hwmcc07_amba_naive.txt
│   ├── neuropdr_no1.log
│   ├── neuropdr_no1_detail.log
│   └── nusmv.syncarb5^2.B.txt
├── model # Model saved after training
│   ├── neuropdr_no1_best.pth.tar
│   └── neuropdr_no1_last.pth.tar
├── requirements.txt # In order to reproduce the experiment environment fast
└── script
    └── fetch_info_from_log.py # Generate table from logs
```

dataset folder (hwmcc 07): /data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/eijk.S953.S

dataset folder (hwmcc 20): /data/hongcezh/clause-learning/data-collect/hwmcc20-7200-result/output

aiger size: /data/hongcezh/clause-learning/data-collect/stat/