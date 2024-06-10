# Extending existing PyTorch container with EasyBuild
- If complex installs are needed, the best practice is to extend the existing container that contains correct bindings, plugins, and environment variables
- This is especially important in distributed training
## Installing the Poro module
- This EasyBuild recipe is intended to use with [this M-DS fork](https://github.com/rakseli/Megatron-DeepSpeed), that is used in continued pretraining of Poro
1. Clone the repo
2. Set up the installation location
```bash
module --force purge
export EBU_USER_PREFIX=$HOME/EasyBuild
```
3. _Add it to your `.profile` or `.bashrc` &rarr; enable  `LUMI` modules to find software installed_
    - _Only needed for custom path instals, evertyhing should work just fine when installing into `$HOME`_
4. Load needed module
```bash
module load LUMI partition/container EasyBuild-user
```
5. Install the Poro module
```
eb recipes/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-deepspeed-v2.1.0-poro.eb
```
6. Check function:
- List the Python packages in the container
```bash
module load CrayEnv
module load PyTorch/2.1.0-rocm-5.6.1-python-3.10-singularity-20240607
singularity exec $SIF bash -c '$INIT_CONDA_VENV ; pip list'
```
## Installation from scratch and making own module with extensions
### Select installation location
- If you participate multiple projects you'll either have personal setup in `$HOME` or a setup in the project `scratch` or `project` dir
- I recommend setting up the installation into `$HOME` &rarr; for some reason, installation in `scratch` resulted `lmod` couldn't find the module even with the right environment variables
- Only one software setup can be loaded at a time, but it can be changed easily changed
### Prerequisite
1. Set up the installation location
```bash
module --force purge
export EBU_USER_PREFIX=$HOME/EasyBuild
```
2. Add it to your `.profile` or `.bashrc` &rarr; enable  `LUMI` modules to find software installed
### Poro
1. Load needed module
```bash
module load LUMI partition/container EasyBuild-user
```
2. Copy a recipe with environment variables and the possibility to extend
```bash
eb --copy-ec <module_you_want_to_extend> ./recipes/
```
 - List of available containers can be found [here](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#singularity-containers-with-modules-for-binding-and-extras), select one with possibility add extra packages
3. Change the variables
- Replace
    - `local_sif = <sif_you_are_using>
    - `local_c_rocm_version=<sif_rocm_version>
    - `local_c_PyTorch_version = <sif_torch_version>
    - `local_dockerhash=<sif_docker_hash>
- Remove
    - `local_c_flashattention_version`
    - `local_c_xformers_version`
    - mentions of unused variables
- Modify
    -  Variables in `local_runscript_python_distributed`
        - the `conda-python-distributed` will use these
        - comment out `#export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`
        - add `export LOCAL_RANK=\$SLURM_LOCALID`
    - other runscripts if needed
4. Add `local_pip_requirements` variable to recipe
```
local_pip_requirements = """
transformers
numpy
parameterized
pybind11
regex
six
tensorboard
black==21.4b0
isort>=5.5.4
packaging

"""
```
- The empty line needs to be at the end

5. Add lines to `postinstallcmds` after the last comma
```
f'cat >%(installdir)s/user-software/venv/requirements.txt <<EOF {local_pip_requirements}EOF',
 f'singularity exec --bind {local_singularity_bind} --bind %(installdir)s/user-software:/user-software %(installdir)s/{local_sif} bash -c \'source /runscripts/init-conda-venv ; cd /user-software/venv ; pip install -r requirements.txt\'',     
 '%(installdir)s/bin/make-squashfs',
 '/bin/rm -rf %(installdir)s/user-software',
```
6. Install the software
```bash
eb recipes/<your_modified_recipe>.eb
```
7. After a clean install check what LUMI stack is needed
```bash
module spider PyTorch/<your_module_name_check_from_SW_in_EBU_USER_PREFIX>
```
8. Check function:
- List the Python packages in the container
- right module name can be found in `$EBU_USER_PREFIX/EasyBuild/SW/container/PyTorch`
```bash
module load CrayEnv
module load PyTorch/<module_name>
singularity exec $SIF bash -c '$INIT_CONDA_VENV ; pip list'
```

- Distributed (just a naive example to show how `conda-python-distributed` is used)
```bash
#!/bin/bash -e
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --mem=480G
#SBATCH --time=00:10:00
#SBATCH --account=project_<your_project_id>

module load CrayEnv  # Which version doesn't matter; it is only to get the container.
module load PyTorch/2.0.1-rocm-5.5.1-python-3.10-singularity-20240404

# Optional: Inject the environment variables for NCCL debugging into the container.   
# This will produce a lot of debug output!     
export SINGULARITYENV_NCCL_DEBUG=INFO
export SINGULARITYENV_NCCL_DEBUG_SUBSYS=INIT,COLL

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

cd mnist
srun --cpu-bind=mask_cpu:$MYMASKS \
 singularity exec $SIFPYTORCH \
    conda-python-distributed -u mnist_DDP.py --gpu --modelpath model
```
- Single node
```bash
salloc -N1 -pstandard-g -t 30:00
module load CrayEnv PyTorch/2.0.1-rocm-5.5.1-python-3.10-singularity-20240404
srun -N1 -n1 --gpus 8 singularity exec $SIF conda-python-simple \
 -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
exit
```