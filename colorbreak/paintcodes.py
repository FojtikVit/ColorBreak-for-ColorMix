"""Encode / decode PrusaSlicer per-triangle painting codes.

PrusaSlicer stores the multi-material painting of a mesh as one hexadecimal string
per painted triangle (``Metadata/Slic3r_facets_annotation.json``, key ``dividing``).
The string is the TriangleSelector bitstream written most-significant nibble first,
i.e. the *last* character of the string is the first nibble of the stream.

Each nibble holds 4 bits, least-significant bit first::

    bits 0-1 (yy): number of split sides (0 => leaf triangle)
    bits 2-3 (xx): leaf: state 0-2, or 3 = "extended state follows"
                   non-leaf: "special side"

Extended leaf state: next nibble ``zzzz``; if ``zzzz != 0b1110`` the state is
``zzzz + 3`` (3..16), otherwise the next two nibbles form 8 bits (LSB first) and the
state is ``value + 17`` (17..255).

A non-leaf triangle with ``yy`` split sides is followed by ``yy + 1`` child codes,
recursively.  The triangle state equals the extruder id (physical 1..N or virtual
id > N), state 0 = unpainted / default.

See ``TriangleSelector::serialize`` / ``deserialize`` and
``FacetsAnnotation::get_triangle_as_string`` in the PrusaSlicer sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List

EXTENDED_MARKER = 0b1110


@dataclass
class Node:
    """One triangle of the splitting tree.  Leaves have ``state``; inner nodes have children."""

    split_sides: int = 0
    special_side: int = 0
    state: int = 0
    children: List["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.split_sides == 0

    def leaves(self) -> Iterator["Node"]:
        if self.is_leaf:
            yield self
        else:
            for c in self.children:
                yield from c.leaves()

    def leaf_states(self) -> List[int]:
        return [leaf.state for leaf in self.leaves()]


def _nibbles(code: str) -> List[int]:
    """Nibble sequence in stream order (reverse of the printed string)."""
    out = []
    for ch in reversed(code):
        v = int(ch, 16)
        out.append(v)
    return out


def decode(code: str) -> Node:
    """Decode a hex code string into a splitting tree."""
    nibs = _nibbles(code)
    pos = 0

    def nxt() -> int:
        nonlocal pos
        if pos >= len(nibs):
            raise ValueError(f"truncated painting code {code!r}")
        v = nibs[pos]
        pos += 1
        return v

    def rec() -> Node:
        c = nxt()
        split_sides = c & 0b11
        xx = c >> 2
        if split_sides:
            node = Node(split_sides=split_sides, special_side=xx)
            node.children = [rec() for _ in range(split_sides + 1)]
            return node
        if xx != 3:
            return Node(state=xx)
        z = nxt()
        if z != EXTENDED_MARKER:
            return Node(state=z + 3)
        v = nxt() | (nxt() << 4)
        return Node(state=v + 17)

    root = rec()
    if pos != len(nibs):
        raise ValueError(f"trailing nibbles in painting code {code!r}")
    return root


def encode(root: Node) -> str:
    """Encode a splitting tree back into PrusaSlicer's hex string."""
    nibs: List[int] = []

    def rec(node: Node) -> None:
        if node.is_leaf:
            n = node.state
            if n < 3:
                nibs.append((n << 2))
            elif n <= 16:
                nibs.append(0b1100)
                nibs.append(n - 3)
            elif n <= 255:
                nibs.append(0b1100)
                nibs.append(EXTENDED_MARKER)
                v = n - 17
                nibs.append(v & 0xF)
                nibs.append(v >> 4)
            else:
                raise ValueError(f"state {n} out of range")
        else:
            nibs.append(node.split_sides | (node.special_side << 2))
            for c in node.children:
                rec(c)

    rec(root)
    return "".join("0123456789ABCDEF"[v] for v in reversed(nibs))


def leaf_states(code: str) -> List[int]:
    """Fast path: just the list of leaf states of a code."""
    return decode(code).leaf_states()


def remap(code: str, mapping: dict) -> str:
    """Return the code with every leaf state ``s`` replaced by ``mapping.get(s, s)``."""
    root = decode(code)
    for leaf in root.leaves():
        leaf.state = mapping.get(leaf.state, leaf.state)
    return encode(root)


def required_version(states) -> int:
    """Painting format version needed for the given set of used states (1 or 2)."""
    return 2 if any(s >= 17 for s in states) else 1


def leaf_code(state: int) -> str:
    """Code of an unsplit triangle in the given state."""
    return encode(Node(state=state))
