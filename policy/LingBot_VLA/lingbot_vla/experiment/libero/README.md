# Install official LIBERO

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git libero # (here)
cd libero
pip install -e .

cd experiment/libero/libero
pip install -r req.txt
```

If can not import xxx from libero.libero please add the libero (here) path to the PYTHONPATH variable.

The results will be save to /project_root/Libero

- release_ensemble/ stores the log files (This directory can be changed by --local_log_dir variable)
- rollouts stores the videos

