"""Exact spoken-word times from the voice job (pure Python, no bpy)."""
import re

class Words:
    def __init__(self, path):
        txt = open(path).read()
        self.w = []
        for blk in re.split(r"\n\s*\n", txt.strip()):
            ls = blk.strip().splitlines()
            if len(ls) < 3: continue
            a, b = ls[1].split(" --> ")
            self.w.append((self._t(a), self._t(b), ls[2].strip()))
        self.end = self.w[-1][1] if self.w else 0
    @staticmethod
    def _t(s):
        h, m, r = s.replace(",", ".").split(":"); return int(h) * 3600 + int(m) * 60 + float(r)
    @staticmethod
    def _n(s): return re.sub(r"[^a-z0-9']", "", s.lower())
    def find(self, phrase, after=0.0, nth=1):
        """index of the first word of `phrase` starting at or after time `after`"""
        p = [self._n(x) for x in phrase.split()]
        hits = 0
        for i in range(len(self.w)):
            if self.w[i][0] + 1e-3 < after: continue
            if all(i + k < len(self.w) and self._n(self.w[i + k][2]) == p[k] for k in range(len(p))):
                hits += 1
                if hits == nth: return i
        raise KeyError(f"phrase not found: {phrase!r} after {after:.2f}")
    def at(self, phrase, after=0.0, nth=1, lead=0.12):
        """time to cut TO a shot for this phrase: just before the first word is spoken"""
        return max(0.0, self.w[self.find(phrase, after, nth)][0] - lead)
    def end_of(self, phrase, after=0.0, nth=1):
        i = self.find(phrase, after, nth); return self.w[i + len(phrase.split()) - 1][1]


    def seq(self, phrases, lead=0.12):
        """cue times for phrases in spoken order: each one is searched after the previous one"""
        out, t = {}, 0.0
        for p in phrases:
            i = self.find(p, t); out[p] = max(0.0, self.w[i][0] - lead); t = self.w[i][0] + 1e-3
        return out
