"""Questions about images to a vision-language model."""


class VisionQuestions:
    def query_vlm(self, images, question):
        """Ask the vision model a question about one or more RGB arrays."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        return self._vision().ask(list(images), question)
