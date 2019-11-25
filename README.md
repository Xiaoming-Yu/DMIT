# DMIT

Pytorch implementation of our paper: ["Multi-mapping Image-to-Image Translation via Learning Disentanglement"](https://arxiv.org/abs/1909.07877).



## Dependencies
you can install all the dependencies  by
```
pip install -r requirements.txt
```
 
## Getting Started

### Datasets
Coming soon...

### Training
- Season Transfer
	```
	bash ./scripts/train_season_transfer.sh
	```
- To view training results and loss plots, run python -m visdom.server and click the URL http://localhost:8097. More intermediate results can be found in environment `exp_name`.

### Testing
- Run
	```
	bash ./scripts/test_season_transfer.sh
	```
- The testing results will be saved in `checkpoints/dmit_season_transfer/results` directory.




#### bibtex
If this work is useful for your research, please consider citing :
```
@inproceedings{yu2019multi,
  title={Multi-mapping Image-to-Image Translation via Learning Disentanglement},
  author={Yu, Xiaoming and Chen, Yuanqi and Liu, Shan and Li, Thomas and Li, Ge},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2990--2999},
  year={2019}
}
 ```
### Acknowledgement
The code used in this research is inspired by [BicycleGAN](https://github.com/junyanz/BicycleGAN), [MUNIT](https://github.com/NVlabs/MUNIT), [DRIT](https://github.com/HsinYingLee/DRIT), [AttnGAN](https://github.com/taoxugit/AttnGAN), and [SingleGAN](https://github.com/Xiaoming-Yu/SingleGAN).
### Contact
Feel free to reach me if there is any questions (xiaomingyu@pku.edu.cn).




