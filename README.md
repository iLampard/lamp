# Event Prediction using Large Language Models


PyTorch code for the paper [Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning](https://arxiv.org/abs/2305.16646).


## How to Run

### Environment Requirements

First, please make sure you have an environment compatible with the following requirement 

```bash
torch == 1.9.0
numpy
pandas
```

Lower version of pytorch should also be working but we have not tested it.



### Data Preparation

You can obtain the benchmark dataset from [Google Drive](https://drive.google.com/file/d/1XbPiPTNVprKaQwMvk9McaY2USJrEkUj6/view?usp=share_link). All the datasets are well pre-processed and can be used easily.

**Please unzipped the files and put gdelt folder under the `./data/` directory**, so the directory becomes `./data/gdelt/*.pkl`. 


### Training and Evaluation Example

Assume we are running the task of predicate (relation) prediction on the GDELT data and setup the config files.


Step 1: we need to train the chosen TPP, ANHP, with the config `configs/ke_anhp_gdelt.yaml`:

```
python main_basemodel.py
```
NOTE: in `configs/ke_anhp_gdelt.yaml`, one needs to setup data and model specs, where we have put default params there.

After the training is finished, the prediction result of the base model will be saved as `logs/ke_anhp_gdelt_test.pkl`.


Step 2: we query the chatgpt to generate the causal events based on the prediction results from the previous step

```
cd scripts/gdelt
python step_4_query_chatgpt.py
```
NOTE: in `scripts/gdelt/step_4_query_chatgpt.py`, one needs to setup the personal openai account to query the gpt, along with the params of the prediction task of predicate (relation).

After the query is finished, a json file `relation.json` will be generated at `scripts/gdelt/ddb_storage/gdelt_chatgpt`.


Step 3: we setup the samples to train the ranking model:

```
cd scripts/gdelt
python step_5_make_emb_dataset.py
```
After the generation is finished, a json file `relation.json` will be generated at `scripts/gdelt/ddb_storage/ke_anhp_gdelt_bert_ebm_dataset`.


Step 4: we train the ranking model, with the config `configs/ke_anhp_gdelt_ebm_rel.yaml` and evaluate on the test set.

```
python main_ebm.py
```

For other tasks, one can simply modify the params in previous steps.


## Reference

If you use this code as part of any published research, please acknowledge the following paper 
```
@misc{shi2023language,
      title={Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning}, 
      author={Xiaoming Shi and Siqiao Xue and Kangrui Wang and Fan Zhou and James Y. Zhang and Jun Zhou and Chenhao Tan and Hongyuan Mei},
      year={2023},
      eprint={2305.16646},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
