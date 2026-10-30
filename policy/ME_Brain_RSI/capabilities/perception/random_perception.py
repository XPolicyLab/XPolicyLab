"""ANR14C: two independent digit readers, pad-count verified.

ANR13C eval15 (random fonts, 15 official layouts): the ANS1 reader was right on
11/15 digit sets, the standard colour-family reader on a different subset
(seed0 L29/L6, seed1 L19 right where ANS1 added or dropped a digit).  Each
candidate set is accepted only when the pad detector finds exactly that many
pads (ANS1 border fit, then the standard circle fit); the ANS1 set keeps
priority.  Offline on those 15 first frames: ANS1 alone 6 correct, combined 9.
"""
from .number_perception import NumberPerception
from .standard_perception import StandardNumberPerception


class RandomNumberPerception(NumberPerception):
    def __init__(self, calibration):
        super().__init__(calibration)
        self.last_diagnostics = {}

    @staticmethod
    def _complete(digits):
        return len(digits) in (4, 5) and len({d['digit'] for d in digits}) == len(digits)

    def _pads(self, rgb, count):
        pads = NumberPerception.pads_from_rgb(self, rgb, count)
        if len(pads) != count:
            pads = StandardNumberPerception.pads_from_rgb(self, rgb, count)
        return pads if len(pads) == count else []

    def detect(self, rgb):
        readers = [('ans1_number_perception', NumberPerception.digits_from_rgb(self, rgb)),
                   ('standard_colour_families', StandardNumberPerception.digits_from_rgb(self, rgb))]
        attempts = []
        for name, digits in readers:
            pads = self._pads(rgb, len(digits)) if self._complete(digits) else []
            attempts.append(dict(reader=name, digits=sorted(d['digit'] for d in digits), pads=len(pads)))
            if pads:
                self.last_diagnostics = dict(source=name, digit_count=len(digits), attempts=attempts,
                                             families=sorted({d.get('family') for d in digits}))
                return digits, pads
        digits = readers[0][1]
        self.last_diagnostics = dict(source='abstain', digit_count=len(digits), attempts=attempts)
        return digits, []
