# GPT2 Prompt for Cross Domain Slot Filling
Using GPT2 prompt learning proposed in Findings of ACL 2022 paper: [Inverse is Better! Fast and Accurate Prompt for Few-shot Slot Tagging](https://arxiv.org/pdf/2204.00885.pdf) for cross domain slot filling task.


Some code in this repository is based on the excellent open-source project: [https://github.com/AtmaHou/PromptSlotTagging](https://github.com/AtmaHou/PromptSlotTagging)


# Requirements
```
python==3.7.11
torch==1.8.1
transformers==4.21.0
seqeval==1.2.2
CUDA==11.1
```

# Experiment
I conduct all experiments on a single 2080Ti gpu. Batch_size is set to 16, and train the model for 10 epochs, all the other hyperparameters are same to the original paper[1]. For all experiments, i take the model which works best on the dev set and evaluate it on the test set.

# Result

|  | 0-samples | 20-samples | 50-samples |
| :-----| ----: | :----: | :----: |
| AddToPlaylist | 40.28 | 60.17 | 69.68 |
| BookRestaurant | 34.29 | 66.84 | 70.49 |
| GetWeather | 57.16 | 65.96 | 73.86 |
| PlayMusic | 18.52 | 49.92 | 61.44 |
| RateBook | 7.39 | 82.84 | 88.99 |
| SearchCreativeWork | 32.24 | 37.53 | 51.81 |
| SearchScreeningEvent | 12.76 | 71.75 | 83.38 |
| **Average F1** | 28.95 | 62.14 | 71.38 |

# Reference
[1] Hou, Yutai, et al. "Inverse is Better! Fast and Accurate Prompt for Few-shot Slot Tagging." Findings of the Association for Computational Linguistics: ACL 2022. 2022.