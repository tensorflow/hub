# Image Retraining Sample

**Topics:** Tensorflow 2.0, TF Hub, Cloud TPU

## Specs
### Cloud TPU

**TPU Type:** v2.8
**Tensorflow Version:** Nightly

### Cloud VM

**Machine Type:** n1-standard-2
**OS**: Debian 9
**Tensorflow Version**: Came with tf-nightly. Manually installed Tensorflow 2.0 Beta

Launching Instance and VM
---------------------------
- Open Google Cloud Shell
- `ctpu up -tf-version nightly`
- If cloud bucket is not setup automatically, create a cloud storage bucket
with the same name as TPU and the VM
- enable HTTP traffic for the VM instance
- SSH into the system
  - `pip3 uninstall -y tf-nightly`
  - `pip3 install -r requirements.txt`
  - `export CTPU_NAME=<common name of the tpu, vm and bucket>`

Chaning Tensorflow Source Code For Support to Cloud TPU:
--------------------------------------------------------
TPU is not Officially Supported for Tensorflow 2.0, so it is not exposed in the Public API.
However in the code, the python files containing the required modules are imported explicitly.
There's a small bug in `CrossShardOptimizer` which tries to use OptimizerV1 and all Optimizers
available in the Public API are in V2. To support V2 Optimizers, a small Code Fragment is needed
to be changed in CrossShardOptimizer's `apply_gradients(...)` function.
To do that
- Browse (`cd`) to the installation directory of tensorflow. 

**To find the installation directory:**
```python3
>>> import os
>>> import tensorflow as tf
>>> print(os.path.dirname(str(tf).split(" ")[-1][1:]))
```

- `cd` to `python/tpu` inside the installation directory
- open `tpu_optimizer.py` in an editor
- change line no. 173 (For Tensorflow 2.0 Beta)
**From**
```python3
     return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)
```
**To**
```python3
     return self._opt.apply_gradients(summed_grads_and_vars, name=name)
```
- Save Changes

Running Tensorboard:
----------------------
### Pre Requisites
```bash
$ sudo -i
$ pip3 uninstall -y tf-nightly
$ pip3 install tensorflow==2.0.0-beta0
$ exit
```

### Launch
```bash
$ sudo tensorboard --logdir gs://$CTPU_NAME/model_dir --port 80 &>/dev/null &
```
To view Tensorboard, Browse to the Public IP of the VM Instance

Running the Code:
----------------------
```bash
$ python3 image_retraining_tpu.py --tpu $CTPU_NAME --use_tpu \
--model_dir gs://$CTPU_NAME/model_dir \
--data_dir gs://$CTPU_NAME/data_dir \
--batch_size 16 \
--iterations 4 \
--dataset horses_or_humans
```
