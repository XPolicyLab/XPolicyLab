"""Benchmarks: one package per benchmark, holding its simulator worker (served with
`harness serve --factory benches.<name>.runtime:create_environment`), task lists,
development layouts and evaluation helpers.

Simulator imports stay inside the worker process; everything a kernel or driver
imports from here must load without Isaac, SAPIEN or torch.
"""
