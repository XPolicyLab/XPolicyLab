from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAME_DE = 9.0
HOLE_DE = 16.0
MIN_HOLE_PX = 25
PRESENT_FRAC = 0.55
GONE_FRAC = 0.55
REST_DEV = 0.02
MOVED_DEV = 0.15
STILL_DE = 4.0
LONG_STILL_DE = 1.0
LONG_STILL_DEV = 0.25
FREE_DELAY = 10
JOIN_DT = 30
BARE_PERSIST = 6


def lab_image(img: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt(np.sum(d * d, axis=-1))


@dataclass
class Hole:
    mask: np.ndarray
    area: int
    centroid: tuple
    last_present: int = -1
    first_gone: int = -1
    present: list = field(default_factory=list)
    gone: list = field(default_factory=list)

    @property
    def left_at(self) -> float:
        if self.first_gone >= 0:
            return float(self.first_gone)
        return float(self.last_present) + 0.5

    def as_dict(self) -> dict:
        return {
            "area": self.area,
            "centroid": [round(float(c), 1) for c in self.centroid],
            "last_present": self.last_present,
            "first_gone": self.first_gone,
        }


class DemoWatch:
    def __init__(
        self,
        img0: np.ndarray,
        step0: int,
        region: np.ndarray,
        exclude: np.ndarray,
        rest_band: np.ndarray,
        v0: int,
        v1: int,
        every: int = 2,
        jpeg_quality: int = 92,
    ):
        self.v0, self.v1 = int(v0), int(v1)
        self.every = max(1, int(every))
        self.q = int(jpeg_quality)
        self.img0 = np.ascontiguousarray(img0[self.v0 : self.v1]).copy()
        self.lab0 = lab_image(self.img0)
        self.region = (region & ~exclude)[self.v0 : self.v1]
        rest = rest_band & ~exclude
        rv, ru = np.nonzero(rest)
        if len(rv):
            self.rest_box = int(rv.min()), int(rv.max()) + 1, int(ru.min()), int(ru.max()) + 1
        else:
            self.rest_box = 0, 1, 0, 1
        r0, r1, c0, c1 = self.rest_box
        self.rest = rest[r0:r1, c0:c1]
        self.rest_lab0 = lab_image(img0[r0:r1, c0:c1])[self.rest]
        self.frames: list[tuple[int, bytes]] = []
        self.step0 = int(step0)
        self._prev_rest = self.rest_lab0
        self.left_rest = False
        self.left_at: int | None = None
        self.back_since: int | None = None
        self.still_since: int | None = None

    def update(self, img: np.ndarray, step: int) -> None:
        import cv2

        step = int(step)
        if (step - self.step0) % self.every == 0:
            ok, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(np.ascontiguousarray(img[self.v0 : self.v1]), cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.q],
            )
            if ok:
                self.frames.append((step, buf.tobytes()))
        r0, r1, c0, c1 = self.rest_box
        lab = lab_image(img[r0:r1, c0:c1])[self.rest]
        dev = float(np.mean(delta_e(lab, self.rest_lab0) > SAME_DE)) if len(lab) else 0.0
        still = float(np.mean(delta_e(lab, self._prev_rest))) if len(lab) else 0.0
        self._prev_rest = lab
        if dev >= MOVED_DEV:
            if not self.left_rest:
                self.left_at = step
            self.left_rest = True
        if self.left_rest and dev <= REST_DEV and (still <= STILL_DE):
            if self.back_since is None:
                self.back_since = step
        else:
            self.back_since = None
        if self.left_rest and dev <= LONG_STILL_DEV and (still <= LONG_STILL_DE):
            if self.still_since is None:
                self.still_since = step
        else:
            self.still_since = None

    def still_for(self, step: int) -> int:
        if self.still_since is None:
            return 0
        return int(step) - self.still_since + 1

    def back_for(self, step: int) -> int:
        if self.back_since is None:
            return 0
        return int(step) - self.back_since + 1

    def changed(self, img_final: np.ndarray) -> np.ndarray:
        import cv2

        lab_f = lab_image(np.ascontiguousarray(img_final[self.v0 : self.v1]))
        ch = (delta_e(lab_f, self.lab0) > HOLE_DE) & self.region
        m = ch.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return (m > 0) & self.region

    def pixel_runs(self, img_final: np.ndarray, band_mask: np.ndarray, min_run: int = 12):
        import cv2

        lab_f = lab_image(np.ascontiguousarray(img_final[self.v0 : self.v1]))
        idx = np.nonzero(band_mask)
        n = len(idx[0])
        rf = lab_f[idx]
        seq = [(self.step0, self.img0)] + [(s, None) for s, _ in self.frames]
        bufs = [None] + [b for _, b in self.frames]
        ref = self.lab0[idx].copy()
        start = np.full(n, self.step0, np.int32)
        last_seen = np.full(n, self.step0, np.int32)
        obj_end = np.full(n, -1, np.int32)
        obj_val = np.zeros((n, 3), np.float32)
        best_len = np.zeros(n, np.int32)
        best_val = np.zeros((n, 3), np.float32)
        gone = np.full(n, -1, np.int32)
        ever_bare = np.zeros(n, bool)
        streak = np.full(n, -1, np.int32)
        t_free = (self.left_at if self.left_at is not None else self.step0) + FREE_DELAY

        def close_runs(sel, end_step):
            sidx = np.nonzero(sel)[0]
            length = end_step - start[sidx] + self.every
            notbare = delta_e(ref[sidx], rf[sidx]) >= HOLE_DE
            ok = (length >= min_run) & notbare & ~ever_bare[sidx]
            s = sidx[ok]
            obj_end[s] = end_step[ok]
            obj_val[s] = ref[s]
            gone[s] = -1
            longer = length[ok] > best_len[s]
            best_len[s[longer]] = length[ok][longer]
            best_val[s[longer]] = ref[s[longer]]

        for k, (step, _) in enumerate(seq):
            if k == 0:
                continue
            img = cv2.cvtColor(cv2.imdecode(np.frombuffer(bufs[k], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            px = lab_image(img[idx[0], idx[1]][None])[0] if n else np.zeros((0, 3), np.float32)
            same = delta_e(px, ref) < SAME_DE
            changed = ~same
            if changed.any():
                close_runs(changed, last_seen[changed])
                ref[changed] = px[changed]
                start[changed] = step
            last_seen[:] = step
            bare = delta_e(px, rf) < SAME_DE
            streak[~bare] = -1
            streak[bare & (streak < 0)] = step
            persistent = bare & (step - streak >= BARE_PERSIST)
            new = persistent & (gone < 0) & (obj_end >= 0)
            gone[new] = streak[new]
            if step > t_free:
                ever_bare |= persistent
        return idx, obj_end, obj_val, best_val, gone

    def holes(self, img_final: np.ndarray, max_n: int | None = None, gap: int = 40) -> list[Hole]:
        import cv2

        cand = self.changed(img_final)
        idx, obj_end, _, best_val, gone = self.pixel_runs(img_final, cand)
        lab_f = lab_image(np.ascontiguousarray(img_final[self.v0 : self.v1]))
        sh = shadow_like(best_val, lab_f[idx])
        obj_end = np.where(sh, -1, obj_end)
        H, W = img_final.shape[:2]
        tmap = np.full(cand.shape, -1, np.int32)
        tmap[idx] = obj_end
        gmap = np.full(cand.shape, -1, np.int32)
        gmap[idx] = gone
        vmap = np.zeros(cand.shape + (3,), np.float32)
        vmap[idx] = best_val
        ch = tmap >= 0
        m = ch.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        ch &= m > 0
        n, lbl = cv2.connectedComponents(
            cv2.morphologyEx(ch.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), connectivity=8
        )
        parts = []
        for k in range(1, n):
            mk = (lbl == k) & ch
            if mk.sum() < MIN_HOLE_PX:
                continue
            for sel in split_by_time(mk, tmap, gap):
                if sel.sum() >= MIN_HOLE_PX:
                    parts.append(sel)
        info = [self._part(p, tmap, gmap) for p in parts]
        joined = True
        while joined:
            joined = False
            for i in range(len(info)):
                for j in range(i + 1, len(info)):
                    a, b = info[i], info[j]
                    if abs(a["t"] - b["t"]) <= JOIN_DT and np.hypot(a["c"][0] - b["c"][0], a["c"][1] - b["c"][1]) < 40:
                        info[i] = self._part(a["mask"] | b["mask"], tmap, gmap)
                        del info[j]
                        joined = True
                        break
                if joined:
                    break
        out = []
        for d in info:
            full = np.zeros((H, W), bool)
            full[self.v0 : self.v1] = d["mask"]
            h = Hole(
                full,
                int(d["mask"].sum()),
                (float(d["c"][0]), float(d["c"][1]) + self.v0),
                last_present=int(d["t"]),
                first_gone=int(d["g"]),
            )
            h.colours = vmap[d["mask"]]
            out.append(h)
        out.sort(key=lambda h: -h.area)
        return out if max_n is None else out[:max_n]

    @staticmethod
    def _part(mask, tmap, gmap):
        vs, us = np.nonzero(mask)
        g = gmap[mask]
        g = g[g >= 0]
        return {
            "mask": mask,
            "t": float(np.median(tmap[mask])),
            "c": (us.mean(), vs.mean()),
            "g": float(np.median(g)) if len(g) else -1.0,
        }

    def timeline(self, holes: list[Hole], img_final: np.ndarray) -> None:
        import cv2

        if not holes:
            return
        lab_f = lab_image(np.ascontiguousarray(img_final[self.v0 : self.v1]))
        idx = [np.nonzero(h.mask[self.v0 : self.v1]) for h in holes]
        ref0 = [self.lab0[i] for i in idx]
        reff = [lab_f[i] for i in idx]
        for step, buf in self.frames:
            img = cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            lab = lab_image(img)
            for h, i, r0, rf in zip(holes, idx, ref0, reff):
                px = lab[i]
                p = float(np.mean(delta_e(px, r0) < SAME_DE))
                g = float(np.mean(delta_e(px, rf) < SAME_DE))
                h.present.append((step, round(p, 3)))
                h.gone.append((step, round(g, 3)))
        for h in holes:
            last = -1
            for step, p in h.present:
                if p >= PRESENT_FRAC:
                    last = step
            h.last_present = last
            first = -1
            for step, g in h.gone:
                if step > last and g >= GONE_FRAC:
                    first = step
                    break
            h.first_gone = first


SHADOW_TOL = 0.12


def shadow_like(px: np.ndarray, ref: np.ndarray) -> np.ndarray:
    L = np.maximum(px[:, 0], 1.0)
    Lr = np.maximum(ref[:, 0], 1.0)
    c = px[:, 1:] / L[:, None]
    cr = ref[:, 1:] / Lr[:, None]
    r = L / Lr
    return (np.linalg.norm(c - cr, axis=1) < SHADOW_TOL) & (r < 0.97) & (r > 0.2) | (delta_e(px, ref) < 6.0)


def split_by_time(mask: np.ndarray, tmap: np.ndarray, gap: int = 40, bin_steps: int = 10) -> list[np.ndarray]:
    t = tmap[mask]
    if len(t) == 0:
        return []
    lo, hi = int(t.min()), int(t.max())
    nb = max(1, (hi - lo) // bin_steps + 1)
    hist = np.bincount(((t - lo) // bin_steps).astype(int), minlength=nb).astype(float)
    sm = np.convolve(hist, np.ones(3), mode="same")
    order_ = np.argsort(-sm)
    modes = []
    for b in order_:
        if sm[b] < MIN_HOLE_PX:
            break
        c = lo + (b + 0.5) * bin_steps
        if all(abs(c - m) >= gap for m in modes):
            modes.append(c)
    if len(modes) <= 1:
        return [mask]
    modes = np.array(sorted(modes))
    vs, us = np.nonzero(mask)
    near = np.argmin(np.abs(tmap[vs, us][:, None] - modes[None, :]), axis=1)
    out = []
    for k in range(len(modes)):
        m = np.zeros(mask.shape, bool)
        m[vs[near == k], us[near == k]] = True
        out.append(m)
    return out


def order(holes: list[Hole]) -> list[int]:
    return sorted(range(len(holes)), key=lambda i: (holes[i].last_present, holes[i].first_gone))
