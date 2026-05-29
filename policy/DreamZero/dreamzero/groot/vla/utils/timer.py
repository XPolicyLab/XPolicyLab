import time


class ContextTimer:

    def __init__(self, trainer):
        self.last_key = None
        self.trainer = trainer
        self.start_times = {}
        self.key_stack = []

    def with_label(self, key):
        self.last_key = key
        return self

    def __enter__(self):
        self.key_stack.append(self.last_key)  # Push key to stack
        self.start_times[self.last_key] = time.time()  # Start timing for this key
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        key = self.key_stack.pop()  # Pop key from stack
        diff = time.time() - self.start_times[key]
        self.trainer.log({f"{key}_time": diff})
        # print(f"{key}: {diff:.2f} seconds")


if __name__ == "__main__":

    class MockTrainer:
        def log(self, data):
            print("Logging:", data)

    trainer = MockTrainer()
    my_timer = ContextTimer(trainer)

    with my_timer.with_label("outer"):
        time.sleep(1)
        with my_timer.with_label("inner"):
            time.sleep(2)
        with my_timer.with_label("inner"):  # Another inner block
            time.sleep(1)
