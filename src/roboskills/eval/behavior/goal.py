"""BEHAVIOR's BDDL goals in words: `describe(problem)` turns a task's `problem0.bddl` into a deterministic instruction.

BEHAVIOR tasks are BDDL problems (objects, `:init` facts, a `:goal` formula), not sentences. The challenge gives half of
its tasks a sentence; for the others this states the goal formula literally, naming each goal object by its category and
where `:init` puts it (`inroom` rooms, the `ontop`/`inside` support it starts on). It says what must hold, never how.
"""

import re


TEMPLATES = {  # predicate -> (sentence when it must hold, sentence when it must not)
    "inside": ("put {0} inside {1}", "take {0} out of {1}"),
    "ontop": ("put {0} on top of {1}", "take {0} off {1}"),
    "nextto": ("put {0} next to {1}", "keep {0} away from {1}"),
    "under": ("put {0} under {1}", "take {0} from under {1}"),
    "touching": ("make {0} touch {1}", "keep {0} from touching {1}"),
    "attached": ("attach {0} to {1}", "detach {0} from {1}"),
    "overlaid": ("spread {0} over {1}", "take {0} off {1}"),
    "contains": ("make {0} contain {1}", "empty {1} out of {0}"),
    "filled": ("fill {0} with {1}", "empty {1} out of {0}"),
    "covered": ("cover {0} with {1}", "clean {1} off {0}"),
    "toggled_on": ("switch on {0}", "switch off {0}"),
    "open": ("open {0}", "close {0}"),
    "cooked": ("cook {0}", "keep {0} uncooked"),
    "frozen": ("freeze {0}", "keep {0} unfrozen"),
    "real": ("make {0}", "get rid of {0}"),
}


def parse(text):
    """S-expressions of a BDDL text: nested lists of tokens (comments dropped)."""
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack = [[]]
    for t in tokens:
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return stack[0]


def section(problem, key):
    """The `(:key ...)` block of a parsed `(define (problem ...) ...)`."""
    for item in problem[0] if problem and isinstance(problem[0], list) else problem:
        if isinstance(item, list) and item and item[0] == key:
            return item[1:]
    raise KeyError(key)


def category(term):
    """`jar__of__honey.n.01_1` / `?jar__of__honey.n.01` -> "jar of honey"."""
    return re.sub(r"\.n\.\d+(_\d+)?$", "", term.lstrip("?")).replace("__", " ").replace("_", " ")


class Once:
    """A quantified variable's words: `first` on its first mention, `then` after."""

    def __init__(self, first, then):
        self.first, self.then, self.used = first, then, False

    def __call__(self):
        word, self.used = (self.then if self.used else self.first), True
        return word


def place(room):
    return room.replace("_", " ")


class Names:
    """How the instruction names things: by category, with the room `:init` puts them in (or their support's room) and,
    on first mention, the support they start on; a number only when two goal objects share a category."""

    def __init__(self, init, goal_terms):
        self.room, self.support, self.said = {}, {}, set()
        for fact in init:
            if len(fact) == 3 and fact[0] == "inroom":
                self.room[fact[1]] = place(fact[2])
            elif len(fact) == 3 and fact[0] in ("ontop", "inside") and not fact[1].startswith("agent."):
                self.support.setdefault(fact[1], (fact[0], fact[2]))
        insts = {t.lstrip("?") for t in goal_terms if re.search(r"_\d+$", t)}
        cats = [category(t) for t in insts]
        self.numbered = {t for t in insts if cats.count(category(t)) > 1}

    def where(self, inst):
        room = self.room.get(inst) or self.room.get(self.support.get(inst, (None, None))[1])
        return f" in the {room}" if room else ""

    def __call__(self, term, binding=None):
        if binding and term in binding:
            return binding[term]()
        inst = term.lstrip("?")
        name = category(inst) + (f" {inst.rsplit('_', 1)[1]}" if inst in self.numbered else "")
        if inst in self.said or inst.startswith("floor."):
            return f"the {name}{self.where(inst) if inst.startswith('floor.') else ''}"
        self.said.add(inst)
        rel, under = self.support.get(inst, (None, None))
        if under is not None and not under.startswith("floor."):
            return f"the {name} ({'on' if rel == 'ontop' else 'in'} the {category(under)}{self.where(under)})"
        return f"the {name}{self.where(inst)}"

    def start(self, cat):
        """Where every object of category `cat` starts, when `:init` puts them all in one place: " (on the sink in the
        bathroom)", " in the living room" (on its floor), else ""."""
        places = {self.support.get(i) for i in self.support if category(i) == cat}
        if len(places) != 1 or None in places or cat in self.said:
            return ""
        self.said.add(cat)
        rel, under = places.pop()
        if under.startswith("floor."):
            return self.where(under)
        return f" ({'on' if rel == 'ontop' else 'in'} the {category(under)}{self.where(under)})"


def phrase(f, names, binding=None, negate=False):
    """One goal formula in words (a lowercase clause)."""
    head = f[0]
    if head == "and":
        return join([phrase(g, names, binding, negate) for g in f[1:]])
    if head == "or":
        return "either " + " or ".join(phrase(g, names, binding, negate) for g in f[1:])
    if head == "not":
        return phrase(f[1], names, binding, not negate)
    if head in ("forall", "exists", "forn", "forpairs"):
        decl, body = (f[2], f[3]) if head == "forn" else (f[1], f[2])
        var, cat = decl[0], category(decl[2])
        if head == "exists":  # one object of the category, the same one wherever the body names it again
            nested = any(isinstance(g, list) and g and g[0] in ("forall", "forn") for g in [body, *body])
            return phrase(
                body,
                names,
                dict(binding or {}, **{var: Once(f"one {cat}" if nested else f"a {cat}", f"that {cat}")}),
                negate,
            ) + (f" (the same {cat} for all)" if nested else "")
        word = {"forall": f"every {cat}", "forn": f"{f[1][0]} of the {cat}s"}.get(head, f"each {cat}")
        return phrase(body, names, dict(binding or {}, **{var: Once(word + names.start(cat), word)}), negate)
    args = [names(a, binding) for a in f[1:]]
    if head in TEMPLATES:
        return TEMPLATES[head][negate].format(*args)
    return f"make {args[0]} {'not ' if negate else ''}{head.replace('_', ' ')}" + (
        f" {' '.join(args[1:])}" if args[1:] else ""
    )


def join(clauses):
    clauses = [c for c in clauses if c]
    return clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]


def terms(f):
    return [t for g in f for t in (terms(g) if isinstance(g, list) else [g])]


def describe(problem_text):
    """The instruction for a BDDL problem: its goal, as one sentence."""
    problem = parse(problem_text)
    goal, init = section(problem, ":goal")[0], section(problem, ":init")
    text = phrase(goal, Names(init, terms(goal)))
    return text[0].upper() + text[1:] + "."
