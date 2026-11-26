"""The runtime: the model writes programs (driver), a kernel runs them in a session
whose API comes from the skills (session, skills, prompts), guards and controllers
run at every tick (reactive) and a fixed shield checks every command (shield).

The driver and its helpers are standard library; the session and the skills import
numpy and model clients inside the kernel process.
"""
