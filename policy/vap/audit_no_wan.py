"""Fail-closed Python-open audit for this serving path's known Wan assets.

This is not a universal filesystem proof: renamed aliases, inherited file
handles, or native-only access paths are outside this optional audit's scope.
"""
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    attempts = []
    markers = ('/Wan-AI/Wan2.2-TI2V-5B/', 'Wan2.2_VAE.pth',
               'models_t5_umt5-xxl-enc-bf16.pth', 'diffusion_pytorch_model')

    def deny_wan(event, arguments):
        if event != 'open':
            return
        path = str(arguments[0]).replace('\\', '/')
        if any(marker in path for marker in markers):
            attempts.append(path)
            raise AssertionError(f'Wan asset access attempted: {path}')

    os.environ.pop('WALLX_MODEL_ROOT', None)
    os.environ.pop('VAP_WAN22_TOKENIZER_PATH', None)
    sys.path[:0] = os.environ['VAP_SOURCE_PATH'].split(os.pathsep)
    report = {'status': 'RUNNING', 'wallx_model_root_unset': True,
              'wan_tokenizer_path_unset': True, 'attempts': attempts,
              'audit_installed_before_server_import': True}
    output = Path(os.environ['VAP_RUN_DIR']) / 'wan_free_server.json'
    output.write_text(json.dumps(report, indent=2))
    sys.addaudithook(deny_wan)
    print('VAP_WAN_FREE_AUDIT armed; WALLX_MODEL_ROOT unset', flush=True)
    try:
        runpy.run_module('XPolicyLab.policy.vap.server',
                         run_name='__main__', alter_sys=True)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        output.write_text(json.dumps(report, indent=2))
        raise
    report['status'] = 'PASS'
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
