# C3-Action-Anticipation

## Challenge
To submit and participate to this challenge, register at the [Action Anticipation Codalab Challenge](https://codalab.lisn.upsaclay.fr/competitions/702)

## Evaluation Code
This repository contains the official code to evaluate egocentric action anticipation methods on the EPIC-KITCHENS-100 validation set. 

### Requirements
In order to use the evaluation code, you will need to install a few packages. You can install these requirements with: 

`pip install -r requirements.txt`

### Usage
You can use this evaluation code to evaluate submissions on the valuation set in the official JSON format. To do so, you will need to first download the public EPIC-KITCHENS-100 annotations with:

`git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git`

You can then evaluate your json file with:

`python evaluate_anticipation_json_ek100.py path_to_json path_to_annotations`

### Example json file
We provide an example json file which has been generated using our "chance" action anticipation baseline. To evaluate this json, you first need to unzip its archive with:

`unzip action_anticipation_chance_baseline_validation.zip`

After that, you can evaluate the json file with:

`python evaluate_anticipation_json_ek100.py action_anticipation_chance_baseline_validation path_to_annotations`

## RULSTM Baseline Models 
The RULSTM baseline models have been trained and evaluated using the official implementation available at https://github.com/fpv-iplab/rulstm.

### Pre-requisites
To train/validate/test models, you first have to clone the RULSTM repository and install the requirements. Please see https://github.com/fpv-iplab/rulstm for detailed instructions. After following instructions, access the `RULSTM` directory with:

`cd rulstm/RULSTM`

### Models
You can download models with:

`chmod +x scripts/download_models_ek100.sh` 
`./scripts/download_models_ek100.sh` 

Models will be downloaded in the `models/ek100` directory.

### Features
You can download TSN features used to train and evaluate the models on EPIC-KITCHENS-100 with:

`chmod +x scripts/download_data_ek100.sh`
`./scripts/download_data_ek100.sh`

### Training
You can train models on EPIC-KITCHENS-100 with:

`chmod +x scripts/train/anticipation_ek100.sh`
`./scripts/train/anticipation_ek100.sh`

### Validation
You can produce json validation files with:

 * `mkdir -p jsons/ek100`;
 * Anticipation: `python main.py validate_json data/ek100 models/ek100 --modality fusion --task anticipation --json_directory jsons/ek100 --ek100 --num_class 3806 --mt5r`;
 * Early recognition: `python main.py validate_json data/ek100 models/ek100 --modality fusion --task early_recognition --json_directory jsons/ek100 -ek100 --num_class 3806 --mt5r`.

These json files can be evaluated using the code contained in this repository.
 
### Test
You can produce json test files with:

 * RGB branch: `python main.py validate data/ek100 models/ek100 --modality rgb --task anticipation --num_class 3806 --mt5r`;
 * Optical Flow branch: `python main.py validate data/ek100 models/ek100 --modality flow --task anticipation --num_class 3806 --mt5r`;
 * Object branch: `python main.py validate data/ek100 models/ek100 --modality obj --task anticipation --feat_in 352 --num_class 3806 --mt5r`;
 * Complete architecture with MATT: `python main.py validate data/ek100 models/ek100 --modality fusion --task anticipation --num_class 3806 --mt5r`.

These json files can be sent to the leaderboard for evaluation.

