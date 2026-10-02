# GenEval Negation Extension

Extends the GenEval benchmark with negation-focused prompts over attributes and
relations. New prompts reuse GenEval's `evaluation_metadata.jsonl` schema
(`{tag, include, exclude?, prompt}`) and are scored by GenEval's detector-based
`evaluate_images.py` (Mask2Former presence/count + CLIP color + bbox geometry).

## Language

**Statement**: One thing a prompt asserts or denies about the scene, and the unit
everything is counted in — "the mug is red", "the cat is not left of the dog". A
`pos<P>_neg<N>` prompt carries P asserted and N denied statements, so `pos2_neg2`
holds four and is scored on all four. Each has a **polarity** (positive or negated)
and exactly one subject object, which is why a statement, unlike a whole prompt row,
can be attributed to a single failure cause without a tie-break.

The on-disk key stays `slots` — in every prompt JSONL and in every result row's
`metadata` — because renaming it would invalidate all existing prompts and results.
Python identifiers follow the key, not the prose. So `slot` survives as a storage
detail and must not appear in anything a reader sees: reports, figures, table
headers, CSV columns and printed output all say **statement**.
_Avoid_: slot (storage only), clause, condition, requirement, constraint

**Balance floor**: The best spread a single prompt file can reach, fixed by how many
rows it has against how large the vocabulary is. 100 rows draw 100 subjects from 79
objects, so the most even possible file gives 21 objects two turns and 58 one turn —
a **staircase**, never a flat line. "Balanced" therefore means *at the floor*, not
*equal*, and the floor differs per dimension: relation predicates (2–4 per category)
and attribute values (4–10) have small enough vocabularies to reach a true zero,
objects never do. Chasing equality past the floor is not achievable, and reading a
file's unevenness as imbalance without comparing it to the floor is a misreading.
Reader-facing figures therefore draw the floor itself as their reference, so the
comparison is visible without quoting a summary statistic.
_Avoid_: perfectly balanced, uniform (unreachable for objects), even split

**Independent balance**: Every prompt file is balanced against **itself alone**, with no
reference to what other files used. Preferred over balancing each file against the
corpus, because a file balanced to the floor makes the corpus balanced for free — a sum
of near-uniform files is near-uniform — whereas corpus-seeding works the other way,
deliberately skewing individual files so they offset each other, and it requires the
per-category generation runs to coordinate. Independence also lets those runs stay
parallel.
_Avoid_: corpus balance (that is the consequence, not the target), global balance

**Prompt extension**: Growing a prompt file by appending only the shortfall, leaving
every existing row byte-identical. Not the same as regenerating the file at a larger
size: a prompt's row index *is* its image directory name, so a row that shifts silently
re-points already-generated images at a different prompt. Row order is therefore part of
the dataset's identity, not an implementation detail.
_Avoid_: regenerate, resample, refresh (all imply the existing rows may move)

**Counted instance**: A detection the evaluator treats as a real object — one whose
confidence clears the **count threshold**, or, when none does, the single most
confident detection of that class. Deliberately stricter than **presence**, which
keeps a permissive threshold so a faint-but-real object still registers: deciding
*how many* there are needs precision where deciding *whether* one exists needs
recall. Extra detections are typically far less confident than the first (median
0.59 against 0.99), so counting every detection would measure the detector's
duplicate rate rather than what the generator drew. Both the **exactly-one rule**
and the attribute verdict operate on this one set, so they cannot disagree about
what exists.
_Avoid_: detection, box (those are the raw detector output, before this filter),
instance (unscoped)

**Exactly-one rule**: A prompt fails when a requested class has more than one
**counted instance**. Every object in these prompts is asked for singly, so a scene
containing two of them does not depict what was asked, whatever its attributes or
relations. It is a scoring condition, not a data filter — the row is marked
incorrect, nothing is discarded.
_Avoid_: filtering condition (nothing is filtered out), duplicate check, count check
(that is GenEval's separate counting *tag*)

**Over-count**: A failure where a requested class has more than one **counted
instance** — the scene shows two toasters where one was asked for. Kept apart from
an **undetected object** because the generator behaviours differ: one drew nothing,
the other drew too many. Like an undetected object it *pre-empts* the attribute and
relation verdicts rather than competing with them, because with two instances the
statement has no single referent — "the toaster is not white" cannot be judged when
there are two toasters and only one is white. A class has 0, 1, or more than 1
counted instance, so over-count and undetected are mutually exclusive by
construction and need no tie-break.
Classes whose referent is a **single object with separable parts** are exempt, because
a detector legitimately returns several boxes for one of them. `skis` is the only such
class in the COCO vocab: when two are detected they sit within a tenth of a box size of
each other at similar size (91% adjacent), i.e. one pair split in two — unlike `car`,
whose two detections differ eightfold in area and are genuinely two cars. Plural
spelling is *not* the criterion; `scissors` over-counts less than the corpus average.
_Avoid_: duplicate, extra object, miscount (does not say which direction), plural object
(the spelling does not predict this — the referent's structure does)

**Attribute**: A visual property bound to a single object, restricted to the two
families GenEval's detector can verify via CLIP zero-shot classification:
**color** — GenEval's existing ten-colour vocabulary, retained verbatim so the CLIP
colour classifier is unchanged from upstream — and **material** (wooden / metal /
plastic / glass), extracted from the material attributes annotated by **OVAD**
(Bravo et al., 2023). Size is NOT an attribute here. Scored **multi-label**:
SigLIP2 (`ViT-SO400M-14-SigLIP2-378`/webli) produces a per-crop **confusion set** =
argmax plus any value within a small logit margin of it (0.5 for colour, 0 for
material), so a genuinely two-valued object (a black-and-brown mug) carries both.
_Avoid_: feature, property (unscoped)

**Size relation**: A comparative spatial relation between two detected objects —
"bigger than" / "smaller than" (mask pixel-area) and "taller than" / "shorter than"
(mask vertical extent) — scored on SAM 2-refined **mask** geometry, not bounding
boxes. Size lives in the relation taxonomy, not the attribute taxonomy, because it
is inherently a two-object comparison. It measures **apparent** (in-image) size,
which is the only size signal a 2D detector exposes.
_Avoid_: size attribute, scale

**Relation category**: A bucket of relation predicates sharing one evaluator
mechanism, so each category can be scored and reported separately. Provenance is
**per category**, not one source: **directional** predicates are GenEval's own
position vocabulary (`prompts/create_prompts.py`), **near** / **behind** / **in
front of** are frequent Visual Genome spatial predicates, **far from** is the
designed complement of near, and the **comparative** predicates are
introduced here with no published source. Restricted to the four categories a 2D
detector can verify reliably: **directional/projective** (left/right/above/below — offset between mask
centroids), **proximity** (near/far — the **mask gap**, i.e. minimum mask-to-mask
distance, against the mean equivalent-circle radius; a single boundary, so the two
are exact complements with no dead zone), **comparative**
(bigger/smaller/taller/shorter — mask pixel-area / vertical extent), and **depth**
(behind/in front of — **facing surfaces** with a distance-scaled confidence,
.

**All four** categories measure **SAM 2-refined masks** (box-prompted from the
detector), never bboxes. Directional, comparative and
depth keep a dead zone where ambiguity fires no predicate (so a negation passes);
proximity does not. **Topological** (inside/contains/overlapping) and **contact**
(on/on top of/under) were removed: their box/mask heuristics were unreliable.
Interaction verbs (holding/riding/wearing) are also EXCLUDED: no detector-based
evaluator covers them.
_Avoid_: relation type (use category), predicate group

**Mask gap**: The shortest distance between any pixel of one object's mask and any
pixel of another's — 0 when they touch or overlap. Proximity is judged on this, not
on centre-to-centre distance, because two large objects can have distant centres
while nearly touching. Expressed relative to object size so it is resolution-free.
_Avoid_: distance (unscoped), centroid distance (that is a different quantity)

**Facing surfaces**: Each object's own pixels lying within a short window beyond the
**mask gap** of the other object — the two mutually facing edges, wherever they are.
Defined whether or not the objects touch, which is what lets depth be judged for
separated pairs. Depth is measured here and nowhere else.

**In front of / behind**: A depth relation decided on the two objects' **facing
surfaces**: the one whose facing pixels are nearer the camera is in front. The call is
made only when the disparity evidence beats a requirement that **grows with how far
apart the objects are** — touching pairs need almost none, distant pairs need a lot —
so the category abstains rather than guess when the image does not settle it. Nearness comes from the monocular depth model (Depth-Anything-V2),
compared only over the facing surfaces.
_Avoid_: closer/nearer (that is the depth model's raw ordering, not this relation)

**Star dataset**: The relation dataset shape — every relation in a row shares one
subject object A (A→B, A→C…).
_Avoid_: single-anchor, subject-only

**Negated attribute**: An attribute statement whose value the prompt explicitly
denies while the object stays present, e.g. "a bench that is not wooden" /
"a non-wooden bench". Success = the object is detected AND the denied value is
**not in its CLIP confusion set** (so a near-tie on the denied value fails the
negation — the object plausibly has it). There is NO object-absence test:
"without a wooden bench" is always rephrased to a non-wooden bench so the object
still renders. A neg-attribute prompt pairs a plain positive object with one
attribute-negated object: "a photo of a {pos_obj} and a non-{neg_attr} {neg_obj}".
_Avoid_: missing object, excluded object (no prompt ever asks for an absent object;
for a requested object that failed to appear see **Undetected object**)

**Undetected object**: A requested object the evaluator could not find — the
detector returned fewer than `count` instances of the class, whatever the cause
(the generator never drew it, or the detector missed it; the two are not
distinguishable from the results). It is the evaluator's `expected <class>>=<n>,
found <k>` condition. Kept strictly apart from attribute and relation failures,
because an object that is not there cannot have its colour or its relations
judged — so a negation says nothing about the model's handling of negation when
the object is undetected.
_Avoid_: missing object (reserved for the prompt-level notion this benchmark does
not use), absent object, undrawn object (conflates generator with detector)

**Failure cause**: The single bucket a failing prompt row is attributed to, for
reporting. Exactly four, mutually exclusive and exhaustive (every failing row has
one, so they sum to 100%): **undetected object**, **both violated**, **negated
only**, **positive only**. An **undetected object** pre-empts the others only when
no *detected* statement was violated — an object that is not there cannot have its
statements
judged, but a violated negation elsewhere in the same row is still real evidence.
There is deliberately **no precedence between negated and positive**: rows
violating both go to their own bucket rather than being tie-broken into one, since
those two are precisely what the benchmark compares and any tie-break would tune
the headline number toward a hypothesis. "Negated only" is therefore a clean lower
bound on negation-specific failure. Use per-statement blame rates instead when the
question is "which value/predicate is hard" rather than "what breaks".

There is a second, **statement-level** partition for the question "of everything that went
wrong, what kind of thing was it": **undetected object**, **over-count**, **negated
violated**, **positive violated**, **dead-zone failure** — one bucket per *failed statement*, also
exclusive and exhaustive, and needing no tie-break because a statement carries a single
polarity. It excludes statements that are innocent (their row failed elsewhere). The two
partitions answer different questions and must not be mixed in one figure: the row
partition's denominator is failing prompts, the statement partition's is failed
statements.
_Avoid_: error type, reason (that is the evaluator's raw free-text string, one
row of which may name several statements), failure mode

**Dead-zone failure**: A **positive** relation edge that failed because the evaluator
fired **no predicate from that edge's own category** — ambiguous geometry, so it
abstained rather than disagreed. Distinct from a wrong-predicate failure, and it can
only ever be a positive failure: on a negated edge an empty predicate set makes the
negation *pass*. Structurally impossible for attributes (a CLIP confusion set always
contains its argmax) and for **proximity** (a single boundary, so
near and far are exact complements). Judged within the edge's own category, because
`compute_relations` reports every predicate that holds across all four — an unmet
"behind" also reports "near" and "left of", which says nothing about why the depth
edge failed.
_Avoid_: no relation (that is the evaluator's wording for the empty set across ALL
categories, which is a stricter and much rarer condition), abstention, ambiguous

**Element**: The smallest unit a human annotator judges about one image — either an
**object element** ("is there a cup?") or a **condition element** ("is the cup far from
the baseball bat?"). One annotated image carries several. An element is *satisfied* when
the annotator's answer equals the statement's **expected** answer, which is `True` for a
positive statement and `False` for a negated one — so answering "no, it is not near" on a
negated statement satisfies it. Elements are what make a disagreement attributable:
object elements bear on the detector, condition elements on the geometry and attribute
thresholds.
_Avoid_: question, annotation, response (a response is the annotator's raw yes/no, before
it is compared with `expected`)

**Dependency rule**: A condition element counts as unsatisfied for every annotator when
any object it needs is **majority-absent**, whatever those annotators answered about the
condition itself. It keeps "the cup is far from the bat" from being credited on an image
where the majority saw no cup.

**Human label**: The reference verdict for one annotated image: satisfied iff **every**
element's own majority across annotators is satisfied, ties counting as unsatisfied, after
the **dependency rule** is applied. A conjunction over elements, so it is not recoverable
from **human score** by any single cut point — a label of 1 occurs as low as 0.778 while a
label of 0 reaches 0.875.
_Avoid_: ground truth (annotators disagree, so it is a majority reference and not truth),
human accuracy

**Human score**: The mean, over annotators, of the fraction of that annotator's elements
which are satisfied. A continuous companion to **human label**, not a threshold on it.

**Human ceiling**: The highest agreement any evaluator could plausibly reach against the
**human label**, measured by leave-one-annotator-out: each annotator's own conjunction
against the majority of the others. Agreement above a category's ceiling indicates noise
rather than quality, so a category's headroom, not its raw agreement, says whether tuning
is worthwhile.
_Avoid_: inter-annotator agreement (that is the raw pairwise measure; the ceiling is the
leave-one-out form directly comparable with the evaluator's own number)

**Free pass**: A negated statement that passed because its category fired no predicate —
the mirror of a **dead-zone failure**, and unearned: the image was not shown to
satisfy the negation, the evaluator merely could not decide. Not currently
measurable, because the evaluator records a `reason` only for failures, so a passing
edge leaves no trace of *why* it passed. It makes negation look easier than it is in
the three dead-zone-capable categories, so any negated-success claim there is an
upper bound.
_Avoid_: vacuous pass, trivially true (that describes an unchallenging *prompt*, not
an undecidable measurement — see **Challenging negation**)

**Violated statement**: A statement the evaluator named as failing *and* whose own object was
detected — so the attribute or relation itself was genuinely wrong. Narrower than a
**blamed** statement: the evaluator also names an edge when one of its endpoints was
never detected ("missing object for 'cup not left of #0'"), which is an
**undetected object**, not a violation. Conflating the two inflates every relation
predicate's difficulty by 5-15 percentage points and reorders which predicate looks
hardest, so all reporting uses violated. "Blamed" is retained only as the raw
evaluator-named signal it has always been.
_Avoid_: blamed (broader — includes undetected endpoints), failed slot (does not
say whose fault), incorrect statement

**Challenging negation**: A negated statement whose asserted (un-negated) form would
be plausible for the specific objects, so the negation is meaningful rather than
trivially-true nonsense. "a cup that is not on the table" is challenging (a cup
plausibly sits on a table); "a cup that is not flying the airplane" is not.
Plausible spans BOTH the common case AND the **rare-but-possible** (uncommon yet
physically real, e.g. "a zebra to the left of a couch"); it excludes the
physically impossible. The selector is prompted, given the specific object(s), to
choose a relation/attribute that is plausible OR rare-but-possible — never
impossible — so negations stay measurable while covering unusual cases.

For **attributes** the selector is asked for values that are challenging in BOTH
directions, since the same value may be asserted or denied: an object's **canonical**
values (hard to deny — "a non-metal pair of scissors" fights a strong prior) and
**surprising-but-depictable** ones (hard to assert — "a purple broccoli"). Values the
object essentially never has are excluded, because denying them ("a non-wooden pair
of scissors", "a non-metal giraffe") is trivially true — any ordinary photo of the
object already satisfies it, so the prompt measures nothing. An
LLM (Qwen3-30B-A3B) SELECTS the plausible
combination only — which partner object + relation, or which attribute, chosen
from the COCO object list and the category/attribute vocab. It does NOT write the
prompt: the prompt string is then assembled by a rule-based template (PosEval
style). LLM verdicts are cached and seeded for reproducibility. Used for
generation only; evaluation stays detector-based.
_Avoid_: hard negative (overloaded ML term), plausible negation

**Model registry**: The dispatch table in `diffusers_generate.py` mapping a
`--model` key to a pipeline loader + generate function, for the ten benchmark
models (generation/models/registry.py). The registry keeps GenEval's
on-disk output contract (`<outdir>/<idx>/samples/NN.png` + `<idx>/metadata.jsonl`)
so `evaluate_images.py` runs unchanged across all models.
_Avoid_: model zoo, backend

**Negation word**: A surface marker that denies a statement — "not", "non-", "no",
"never" — applied rule-based to the prompt. For relations it mirrors PosEval's
`"not left of"` position encoding; for attributes it scopes to the attribute
only (the object remains present).
_Avoid_: negative prompt (that term means classifier-free-guidance negatives in diffusers)
