"""Model services: the processes skills and agents call (segmenter, pointer, VLMs,
policies, world models), how they are described (registry: services/<name>.toml),
where one machine runs them (deploy: configs/deploy/<machine>.toml), and the tools
that start, stop, check and fetch them (supervisor, tools). Standard library only;
the services themselves run in their own processes.
"""
