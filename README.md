# Extending existing PyTorch containers with EasyBuild
- If additional Python packages are needed, the best practice is to extend the existing container that contains correct bindings, plugins, and environment variables
- This is especially important in distributed training where several people may work with same code
    - Avoid depending personal files and configurations &rarr; one global module for all
    - Users won't break the backend module &rarr; only e.g. data or Python code changes
- If your friendly collegue has already installed a custom module user only needs to set `export EBU_USER_PREFIX="<location>/Easybuild"`
- See [Using the module section](##Using-the-module) for details
- If you want to extend a module yourself, continue reading
# Recipes
- There are currently two example recipes
- `recipes/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-deepspeed-v2.1.0-poro.eb`
    - Example of LUMI provided container extension
- `recipes/lumi-pytorch_2.4.1_lumi_dbg_deepspeed_apex.eb`
    - Example of CSC provided container extension
- LUMI containers use conda based environments and CSC containers use venv based environments
- Fundamentally the recipes are the same but there are few enviroment variable differences
- LUMI containers are run with `conda-python-simple` and `conda-python-distributed`
- CSC containers are run with `python-simple` and `python-distributed`
## Installing premade custom modules
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
5. Install
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
## Using the module


- shortened version of M-DS training
```bash
#!/bin/bash
#SBATCH --job-name=distributed-job
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G # request all memory node has, use explicit number to avoid node failures
#SBATCH --partition=dev-g
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=mi250:8 #reserve full node
#SBATCH --exclusive=user #job allocation don't share nodes with other running jobs
#SBATCH --hint=nomultithread #don't use extra threads with in-core multi-threading which can benefit communication intensive applications
#SBATCH --account=<project>
#SBATCH --output=../training_logs/%x_%j.output
#SBATCH --error=../training_logs/%x_%j.error

export EBU_USER_PREFIX="<location>/Easybuild"
module --force purge
module load LUMI/24.03
module load PyTorch/2.4.1-rocm-6.1.0-apex-1.13.2-deepspeed-0.15.1-20240923

c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

BIND_MASK="$BIND_MASK_1"
CMD=" \
        ../pretrain_gpt.py \
        --tensor-model-parallel-size $TP_SIZE \
        --pipeline-model-parallel-size $PP_SIZE \
        $GPT_ARGS \
        $OUTPUT_ARGS \
        --save $SAVE_CHECKPOINT_DIR \
        --load $LOAD_CHECKPOINT_DIR \
        --train-data-path $TRAIN_DATA_PATH \
        --valid-data-path $VAL_DATA_PATH \
        --data-impl mmap \
        --dataloader-type single \
        --num-workers 2 \
         $DEEPSPEED_ARGS
        "
echo "START $SLURM_JOBID: $(date)"
echo "Python command given: \n"
echo "$CMD"

echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec $SIF \
    python-distributed -u $CMD

echo "END $SLURM_JOBID: $(date)"

```

# Installation from scratch using LUMI provided template and making own module with extensions
- If you don't want to write EasyBuild config from scratch, just modify the `local_pip_requirements` variable in example recipes and follow [Installing premade custom modules](##Installing-premade-custom-modules)
## Select installation location
- If you participate multiple projects you'll either have personal setup in `$HOME` or a setup in the project `scratch` or `project` dir
- I recommend setting up the installation into `$HOME` &rarr; for some reason, installation in `scratch` resulted `lmod` couldn't find the module even with the right environment variables
- Only one software setup can be loaded at a time, but it can be changed easily changed
## Prerequisite
1. Set up the installation location
```bash
module --force purge
export EBU_USER_PREFIX=<desired-location>/EasyBuild
```
### CSC provided containers
1. Load needed module
```bash
module load LUMI partition/container EasyBuild-user
```
2. Copy container to-be-extended same dir as your recipe
```bash
cp /appl/local/csc/soft/ai/images/pytorch_2.4.1_lumi_dbg.sif ./recipes/
```
3. Change variables if needed
4. Modify the distributed script if needed
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
    # Create the virtual environment and space for other software installations that
    # can then be packaged.
    'mkdir -p %(installdir)s/user-software/venv',
    # For the next command, we don't need all the bind mounts yet, just the user-software one is enough.
    #Adding export PYTHONNOUSERSITE="true" allows global enhiretance, not .local
    f'singularity exec --bind %(installdir)s/user-software:/user-software %(installdir)s/{local_sif} bash -c \'cd /user-software/venv ;export PYTHONNOUSERSITE="true"; python -m venv {local_venv} --system-site-packages\'',
    f'cat >%(installdir)s/user-software/venv/requirements.txt <<EOF {local_pip_requirements}EOF',
    f'singularity exec --bind {local_singularity_bind} --bind %(installdir)s/user-software:/user-software %(installdir)s/{local_sif} bash -c \'source /runscripts/init-venv ; cd /user-software/venv ; pip install -r requirements.txt\'',     
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
### LUMI provided containers
1. Load needed module
```bash
module load LUMI partition/container EasyBuild-user
```
2. Copy a recipe and the container into your desired location
```bash
eb --copy-ec <module_you_want_to_extend> ./recipes/
```
 - List of available containers can be found [here](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#singularity-containers-with-modules-for-binding-and-extras), select one with possibility add extra packages
3. Change the variables if needed

4. Modify the distributed script if needed e.g
    -  Variables in `local_runscript_python_distributed`
        - the `conda-python-distributed` will use these
        - comment out `#export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`
        - add `export LOCAL_RANK=\$SLURM_LOCALID`
    - edit other runscripts if needed
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
- In general, the module will be available in all versions of the  LUMI stack and in the CrayEnv stack, but may be worth of checking if you run into problems
8. Check function:
- List the Python packages in the container
- right module name can be found in `$EBU_USER_PREFIX/EasyBuild/SW/container/PyTorch`
```bash
module load CrayEnv
module load PyTorch/<module_name>
singularity exec $SIF bash -c '$INIT_CONDA_VENV ; pip list'
```