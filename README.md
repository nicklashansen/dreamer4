<h1>Dreamer 4 in PyTorch</span></h1>

This is an *unofficial* PyTorch implementation of the Dreamer 4 world model from the paper [*Training Agents Inside of Scalable World Models*](https://arxiv.org/abs/2509.24527) by Danijar Hafner, Wilson Yan, Timothy Lillicrap.

Our implementation is not a complete reproduction of the original work, but we believe that it may serve as a good starting point for anyone looking to extend or experiment with the Dreamer 4 architecture. Notably, the original paper applies Dreamer 4 to Minecraft (discrete actions) whereas this implementation applies it to **multi-task DMControl** (continuous actions). We welcome contributions to improve the codebase and bring it closer to the original paper.

This PyTorch implementation is loosely based on [Edward Hu](https://github.com/edwhu)'s [JAX implementation](https://github.com/edwhu/dreamer4-jax).

----

![Demo](assets/0.gif)

**Try it out!** We provide model checkpoints and a simple web interface for interaction with the world model.

----

## Architecture

![Dreamer 4 Architecture](assets/1.png)

Dreamer 4 consists of a causal tokenizer and an interactive dynamics model, which both use the same block-causal transformer architecture. The tokenizer encodes partially masked image patches and latent tokens, squeezes the latents through a low-dimensional projection with tanh activation, and decodes the patches. It uses causal attention to achieve temporal compression while allowing frames to be decoded one by one. The dynamics model operates on the interleaved sequence of actions, shortcut noise levels and step sizes, and tokenizer representations. It denoises representations via a shortcut forcing objective.

----

## Dataset

<img src="assets/tasks/walker-run.gif" width="14.28%"><img src="assets/tasks/finger-spin.gif" width="14.28%"><img src="assets/tasks/hopper-hop.gif" width="14.28%"><img src="assets/tasks/jumper-jump.gif" width="14.28%"><img src="assets/tasks/walker-run-backward.gif" width="14.28%"><img src="assets/tasks/pendulum-swingup.gif" width="14.28%"><img src="assets/tasks/reacher-hard.gif" width="14.28%"></br>

Our dataset is available for download on [https://huggingface.co/datasets/nicklashansen/dreamer4](HuggingFace) and contains 7,200 mixed-quality trajectories (3.6M frames) spanning **30 continuous control tasks** from [DMControl](https://arxiv.org/abs/1801.00690) and [MMBench](https://arxiv.org/abs/2511.19584). To construct the dataset, we collect 240 trajectories per task using expert [TD-MPC2](https://www.tdmpc2.com) agents that were released as part of our [Newt/MMBench](https://www.nicklashansen.com/NewtWM) project. We use a default resolution of 128×128 for training but the dataset supports up to 224×224. We provide:

| Directory  |  Description  | Trajs. per task | Total trajs.
|--------|---------------|------------|-------|
| [expert](https://huggingface.co/datasets/nicklashansen/dreamer4/tree/main/expert) | Expert demos | 20 | 600 |
| [mixed-small](https://huggingface.co/datasets/nicklashansen/dreamer4/tree/main/mixed-small) | Mixed quality |  20 | 600 |
| [mixed-large](https://huggingface.co/datasets/nicklashansen/dreamer4/tree/main/mixed-large) | Mixed quality |  200 | 6000 |

The data generation pipeline used to generate this dataset is similar to that of our [Newt repository](https://github.com/nicklashansen/newt), except in the case of `mixed-small` and `mixed-large` we add diverse noise patterns to actions to increase diversity of behaviors present in the data. Our released model checkpoints were trained on data from **all three directories**. In the following, we will walk you through data preprocessing, training a **single** world model on **all 30 tasks**, as well as how to interact with it via a simple web interface.

----

## Installation

You will need a machine with at least one CUDA-enabled GPU for inference. If you wish to train the tokenizer or dynamics model yourself we recommend using a machine with >256 GB RAM and 8 GPUs (>24 GB of memory each). We recommend installing dependencies in a virtual environment. We provide an `environment.yaml` file for easy installation with [Conda](https://docs.conda.io/en/latest/):

```
conda env create -f environment.yaml
conda activate dreamer4
```

This should install all required packages. Dataset and model checkpoints are (depending on your use case) optional and can be downloaded separately. Our data is made available [here](https://huggingface.co/datasets/nicklashansen/dreamer4) and model checkpoints are available [here](https://huggingface.co/nicklashansen/dreamer4).

----

## Data preprocessing

Our provided dataset needs to be preprocessed into a sharded data format to ensure efficient data loading during training. Please be aware that the full preprocessed dataset requires approximately **350 GB of disk storage**. To preprocess the data, run the following command:

```
python preprocess_data.py
```

This will preprocess data located in the `FILEDIR` directory (see code comments in `preprocess_data.py` for details) and save the preprocessed shards to the `OUTDIR` directory. You can change these paths by modifying the respective variables in `preprocess_data.py`. To reproduce the results presented in this repo, you will need to preprocess all three data directories provided. Pleases note that the preprocessing step may take several hours depending on your hardware. After preprocessing, you should see a number of shard files in the output directory; repeat this process for all three data directories (`expert`, `mixed-small`, `mixed-large`) to obtain the full dataset for training.

----

## Training the Tokenizer

![Tokenizer training](assets/2.png)

To train the Dreamer 4 tokenizer, first set the ``--data-dirs`` argument in `train_tokenizer.py` to one or more preprocessed data directories, e.g.

```
p.add_argument("--data_dirs", type=str, nargs="+", default=[   # paths to preprocessed frames
        "/<path>/expert-shards",
        "/<path>/mixed-small-shards",
        "/<path>/mixed-large-shards",
    ])
```

and then run the following command:

```
torchrun --nproc_per_node=8 train_tokenizer.py
```

This will start the training process using 8 GPUs. The tokenizer checkpoints will be saved in the `./logs/tokenizer_ckpts/` directory by default; please check the script for a full list of configurable arguments.

You can expect training to take approximately 24 hours on 8× RTX 3090 GPUs, after which you should see curves that look something like this:

<img src="assets/tokenizer/psnr.png" width="33.33%"><img src="assets/tokenizer/loss.png" width="33.33%"><img src="assets/tokenizer/z_std.png" width="33.33%"></br>

----

## Training the World Model

![Dynamics training](assets/3.png)

To train the Dreamer 4 dynamics model with action conditioning, first set the `--data-dirs` and `--frame_dirs` arguments in `train_dynamics.py` to one or more unprocessed and preprocessed data directories, respectively, and run the following command:

```
torchrun --nproc_per_node=8 train_dynamics.py --use_actions
```

This will start the training process using 8 GPUs. Note that this script assumes that you already have a trained tokenizer; if your tokenizer is located at a different path than the default `./logs/tokenizer_ckpts/latest.pt`, you can specify it using the `--tokenizer_ckpt` argument. You can expect training to take approximately 48 hours on 8× RTX 3090 GPUs, after which you should see curves that look something like this:

<img src="assets/dynamics/action_shuffle_ratio.png" width="33.33%"><img src="assets/dynamics/loss.png" width="33.33%"><img src="assets/dynamics/psnr.png" width="33.33%"></br>

----

## Interactive Web Interface

![Web interface](assets/4.gif)

We provide a simple web interface for interaction with the trained world model. The web server runs locally on your machine and requires a CUDA-enabled GPU with at least 2 GB memory. Before starting the web server, verify that all paths in `interactive.py` point to valid directories and checkpoints. You can download our provided checkpoints from [HuggingFace](https://huggingface.co/nicklashansen/dreamer4/tree/main) or train your own. To start the web server, run the following command:

```
python interactive.py
```

and navigate to `http://localhost:7860` in your web browser (assuming you are running it locally and use the default port `7860`). If you are running the web server on a remote (headless) machine, you should still be able to view it locally in your browser after forwarding via SSH, e.g.

```
ssh -L 7860:127.0.0.1:7860 user@remote-machine
```

assuming the web server is running on `remote-machine` and you want to forward it to your local port `7860`. You can customize the port using the `--port` argument.

----

## Checkpoints

Pre-trained tokenizer and dynamics model checkpoints can be downloaded from [HuggingFace](https://huggingface.co/nicklashansen/dreamer4/tree/main). We provide:

| Model  |  Description  | Resolution | Tasks |
|--------|---------------|------------|-------|
| [Tokenizer](https://huggingface.co/nicklashansen/dreamer4/resolve/main/tokenizer_ckpt.pt) | Trained for 90k steps | 128×128 | 30 tasks |
| [Dynamics Model](https://huggingface.co/nicklashansen/dreamer4/resolve/main/dynamics_ckpt.pt) | Trained for 40k steps with actions |  128×128 | 30 tasks |

----

## Citation

If you use this codebase in your research, please consider citing us as:

```
@misc{Hansen2026Dreamer4PyTorch,
    title={Dreamer 4 in PyTorch},
    author={Nicklas Hansen},
    year={2026},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/nicklashansen/dreamer4}},
}
```

as well as the original Dreamer 4 paper:

```
@misc{Hafner2025TrainingAgents,
    title={Training Agents Inside of Scalable World Models}, 
    author={Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year={2025},
    eprint={2509.24527},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2509.24527}, 
}
```

----

## Contributing

You are very welcome to contribute to this project! We especially welcome contributions to improve the codebase and bring it closer to the original paper. Feel free to open an issue or pull request if you have any suggestions or bug reports, but please review our [guidelines](CONTRIBUTING.md) first.

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.

----

## References

- [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) - Danijar Hafner, Wilson Yan, Timothy Lillicrap
- [Learning Massively Multitask World Models for Continuous Control](https://arxiv.org/abs/2511.19584) - Nicklas Hansen, Hao Su, Xiaolong Wang
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) - Nicklas Hansen, Hao Su, Xiaolong Wang
- [Dreamer4 JAX Implementation](https://github.com/edwhu/dreamer4-jax) - Edward Hu
