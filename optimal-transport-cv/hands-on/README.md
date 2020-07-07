# Hands-On Session on "Optimal Transport in Computer Vision"
Welcome to the GitHub directory for the hands-on session on "Optimal Transport in Computer Vision" by Nicolas Courty.
## Offline Version vs Google Colaboratory Version
To overcome the issues related to platforms and resources that are needed to properly execute the scripts contained in the Jupyter Notebooks, we decided to provide the VISUM 2020 participants the possibility to run this code in both offline and online settings.
### Offline Version
To use the offline version of the Notebooks, you need to clone the VISUM 2020 Official Github repository. If you are using Linux or macOS, please open a Terminal window and type:
```bash
$ git clone https://github.com/visum-summerschool/visum-2020.git
```
And then go to this session directory:
```bash
$ cd visum-2020/optimal-transport-cv/hands-on
```
This version also assumes that you have an [Anaconda](https://www.anaconda.com) (or similar) Python environment. Otherwise, assuming that you have, at least, Anaconda in your computer, you can create an environment. In your Terminal window, type:
```bash
$ conda create --name visum2020 python=3.7
```
And then, install all the necessary packages:
```bash
$ pip install -r requirements.txt
```
We recommend the utilization of [Jupyter-Lab](https://jupyter.org/install):
```bash
$ pip install jupyterlab
```
After this, change your Anaconda environment:
```bash
$ conda activate visum2020
```
And run Jupyter-Lab:
```bash
$ jupyter-lab
```
### Google Colaboratory Version
To run the Notebooks on your [Google Drive](https://drive.google.com) you need to use [Google Colaboratory](colab.research.google.com). We have already prepared a ZIP file, that you can download [here](https://filesender.fccn.pt/filesender/?vid=2b7fdf40-2ae3-9749-3a56-000028887046). Decompress the ZIP file and upload the "VISUM 2020" folder to the main directory of your Google Drive. Open **"VISUM 2020/Optimal Transport/Hands-On Google Colaboratory"** folder and run the Notebooks from there.