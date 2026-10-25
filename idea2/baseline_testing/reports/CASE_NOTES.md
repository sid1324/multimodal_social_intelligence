# Midterm case-review notes (in progress)

These cases come only from the team-supplied midterm panel. Descriptions use
the allowed `video_prev.mp4` clip and the model's saved response. They are
observations and hypotheses, not verified explanations of model internals.

## Food stand: correct scene description, wrong final option

ID `82e96128-1242-4e57-8a3e-9ea222df022c_2985-80` (5.0 s preceding
clip). Eight sampled frames span the full clip: a street-food worker prepares
food at a stand and a second worker is nearby. The released best action is
option 3, “Order food from the person at the stand.” The baseline's prose
describes a street-food vendor and explicitly says ordering food is the
appropriate interaction, but its final answer is `1`, “Offer the person a
napkin.” Its selected justification is also `1`. The failure is therefore
visible at the answer-mapping/decision step even though its scene description
supports the released action. A final-answer consistency check is a plausible
future diagnostic; we should not infer that the model's visual perception was
wholly wrong from this item.

## Sidewalk group: apparent label-context mismatch

ID `9d965c16-0014-45e5-b37b-50a786971470_8107-86` (6.37 s preceding
clip). Eight sampled frames span the clip: the wearer walks behind two people
on a sidewalk toward a street crossing. The released best action is option 1,
“Join the group at the restaurant table,” while the model selects option 2,
“Follow the tour guide promptly.” Its prose notes that it cannot see a
restaurant or host in the clip. The visible evidence supports group following
but does not establish a tour guide either. This is a potentially ambiguous
item under a preceding-video-only protocol; inspect the full annotation and
avoid presenting it as a clean visual-reasoning failure. The mismatched-video
control selects the empty option, as expected for visibly unrelated footage,
but that alone does not establish causal grounding.

## Workshop: correct safety rationale truncated before answer

ID `e3f37c38-bf24-481f-b1c6-8d472fab4290_1679-92` (preceding workshop
clip, eight frames across its duration). A worker uses tools amid boards,
power cords, and debris. The released best action is option 1, offering
safety glasses and ear protection. The model's action prose explicitly
advocates that protective gear and rejects less urgent alternatives, but
generation reaches the 192-token cap before a final integer. Native-style
parsing finds no valid choice, and the subsequent justification prompt receives
no selected behavior. This is an output-completion failure with downstream
conditioning effects, not clear evidence that the model failed to recognize
the safety norm. A response-cap or constrained-answer ablation would separate
these mechanisms.

## Group walk: a successful action/justification pair

ID `f0041c6c-06b2-4c3b-b8e6-934305b4226e_6269-35` (5.3 s preceding
clip). Frames show the wearer moving with nearby people and a dog in a grassy
area. The model selects the released option 1, continuing to walk with the
group, and justification 1 about maintaining group cohesion. This is a
successful pair under the benchmark. The clip has no audio, so the
conversation part of the behavior is not independently visible; the correct
selection demonstrates context matching more directly than speech perception.

## Shampoo station: possible temporal-substep error

ID `42d72164-8f4a-49e4-8b95-b3ad423ba65b_138-24` (5.0 s preceding
clip). Frames span the full allowed clip and show a client at a shampoo basin,
with a worker handling a handheld sprayer near already wet hair. The released
best action is option 4, rinsing thoroughly with warm water. The model selects
option 3, applying gentle shampoo, while selecting justification 4, which
supports the released rinse action. Both grooming actions are physically
plausible at this station, but the sprayer and wet hair suggest the rinsing
stage has begun. This is a candidate temporal-state/answer-consistency
failure, not proof of a single internal cause; the 8-frame sample may miss
fine-grained hand movements between frames.

## Salon privacy: RLOO regresses on justification while preserving action

ID `c34effc7-6198-4c85-80c7-c8be85b77dd3_2128-22` (5.0 s preceding
clip). The sampled frames show a stylist working on a client's hair; the
client looks at a phone near the end. The released best action is option 1,
applying product without looking at or discussing the phone, paired with a
privacy justification. The frozen base model selects action 1 and
justification 1. RLOO still selects action 1 but changes to justification 2,
about proactively accommodating the client's ongoing activities. Its
justification prose refers to *engaging* with the phone, contrary to the
selected action. This is a measured paired regression in action/justification
consistency on a privacy item, not evidence of a general Privacy-category
decline. The corresponding frame sheet is retained locally under
`reports/generated/error_review_paired/`; source frames are not redistributed.

On the 400-ID fixed-cap panel, RLOO fixes 27 base joint failures and loses
17 base joint successes; 278 are wrong for both models. This turnover is why
the +10 net correct examples should not be described as a uniform correction
of social-norm failures.
