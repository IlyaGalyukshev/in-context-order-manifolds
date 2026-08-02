# BCS sample stimuli & questions

*Curated, deterministic examples of every stimulus type and question family (regenerate: `python scripts/dump_samples.py`). Full records in `samples.jsonl`.*

## Total order · S1 size · N=9 · easy · shuffle

*The primary design. Interior = ranks 3..N−2. Metric families included.*

**latent order (rank 1..N):** 1:thrindrel < 2:drend < 3:groaft < 4:smeene < 5:spiemb < 6:skine < 7:speerl < 8:crulb < 9:glizzle

**prompt:**
```
The drend is smaller than the crulb.
The smeene is larger than the drend.
The groaft is larger than the thrindrel.
The spiemb is smaller than the speerl.
The drend is smaller than the groaft.
The smeene is smaller than the spiemb.
The glizzle is larger than the speerl.
The speerl is smaller than the crulb.
The crulb is smaller than the glizzle.
The speerl is larger than the skine.
The thrindrel is smaller than the smeene.
The skine is smaller than the glizzle.
The thrindrel is smaller than the drend.
The crulb is larger than the skine.
The skine is larger than the spiemb.
The glizzle is larger than the thrindrel.
The groaft is smaller than the smeene.
The spiemb is larger than the groaft.

Entities: the glizzle, the drend, the speerl, the thrindrel, the smeene, the spiemb, the skine, the crulb, the groaft.
```

**questions (family → key):**
- `reconstruction` — Using only the relations stated above, list all entities from the smallest to the largest (this order may differ from the order the lines appear in). Reply with one entity name per line, nothing else.  **→ ['thrindrel', 'drend', 'groaft', 'smeene', 'spiemb', 'skine', 'speerl', 'crulb', 'glizzle']**
- `reconstruction` — List all entities in the order they first appear in the text above, top to bottom. Reply with one entity name per line, nothing else.  **→ MENTION_ORDER**
- `pairwise` — By the relations above, which is smaller: the spiemb or the skine? Reply with only one entity name. No explanation.  **→ spiemb**
- `pairwise` — By the relations above, which is smaller: the skine or the spiemb? Reply with only one entity name. No explanation.  **→ spiemb**
- `rank` — Counting the smallest as position 1, what is the drend's position in the order? Reply with only the number. No explanation.  **→ 2**
- `rank` — Counting the smallest as position 1, what is the groaft's position in the order? Reply with only the number. No explanation.  **→ 3**
- `betweenness` — Using only the relations stated, which of these is between the other two in the order (counting the smallest as position 1): the skine, the spiemb, or the speerl? Reply with only one entity name. No explanation.  **→ skine**
- `betweenness` — Using only the relations stated, which of these is between the other two in the order (counting the smallest as position 1): the crulb, the thrindrel, or the smeene? Reply with only one entity name. No explanation.  **→ smeene**
- `successor` — Using only the relations stated, which entity is immediately after the spiemb in the order (the next position toward the largest, counting the smallest as position 1)? Reply with only the entity name. No explanation.  **→ skine**
- `successor` — Using only the relations stated, which entity is immediately after the crulb in the order (the next position toward the largest, counting the smallest as position 1)? Reply with only the entity name. No explanation.  **→ glizzle**
- `predecessor` — Using only the relations stated, which entity is immediately before the smeene in the order (the next position toward the smallest, counting the smallest as position 1)? Reply with only the entity name. No explanation.  **→ groaft**
- `predecessor` — Using only the relations stated, which entity is immediately before the groaft in the order (the next position toward the smallest, counting the smallest as position 1)? Reply with only the entity name. No explanation.  **→ drend**
- `count_between` — Using only the relations stated, how many entities are strictly between the skine and the groaft in the order? Reply with only the number. No explanation.  **→ 2**
- `count_between` — Using only the relations stated, how many entities are strictly between the drend and the crulb in the order? Reply with only the number. No explanation.  **→ 5**
- `comparative_distance` — Using only the relations stated, which is closer to the groaft in the order: the smeene or the speerl? Reply with only one entity name. No explanation.  **→ smeene**
- `comparative_distance` — Using only the relations stated, which is closer to the skine in the order: the spiemb or the glizzle? Reply with only one entity name. No explanation.  **→ spiemb**
- `extremes` — Using only the relations stated, which entity is the smallest of all — the one that comes before every other entity in the order? Reply with only the entity name. No explanation.  **→ thrindrel**
- `extremes` — Using only the relations stated, which entity is the largest of all — the one that comes after every other entity in the order? Reply with only the entity name. No explanation.  **→ glizzle**

---

## Total order · S0 zib (symbolic) · N=9 · hard · shuffle

*S0 = arbitrary transitive relation declared in-context (no magnitude lexicon).*

**latent order (rank 1..N):** 1:greemth < 2:sloarl < 3:throamb < 4:flend < 5:pronnel < 6:floane < 7:skind < 8:gleendrel < 9:trostrick

**prompt:**
```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The greemth zibs the floane.
The pronnel is zibbed by the greemth.
The skind zibs the gleendrel.
The sloarl zibs the gleendrel.
The trostrick is zibbed by the pronnel.
The gleendrel zibs the trostrick.
The sloarl zibs the throamb.
The throamb zibs the skind.
The trostrick is zibbed by the greemth.
The greemth zibs the sloarl.
The skind is zibbed by the sloarl.
The throamb zibs the flend.
The pronnel zibs the floane.
The floane is zibbed by the throamb.
The floane zibs the skind.
The gleendrel is zibbed by the flend.
The flend zibs the pronnel.
The flend zibs the trostrick.

Entities: the gleendrel, the flend, the sloarl, the floane, the skind, the pronnel, the greemth, the throamb, the trostrick.
```

**questions (family → key):**
- `reconstruction` — Using only the relations stated above, list all entities from the first to the last (this order may differ from the order the lines appear in). Reply with one entity name per line, nothing else.  **→ ['greemth', 'sloarl', 'throamb', 'flend', 'pronnel', 'floane', 'skind', 'gleendrel', 'trostrick']**
- `reconstruction` — List all entities in the order they first appear in the text above, top to bottom. Reply with one entity name per line, nothing else.  **→ MENTION_ORDER**
- `pairwise` — By the relations above, which is earlier in the zib-order: the floane or the skind? Reply with only one entity name. No explanation.  **→ floane**
- `pairwise` — By the relations above, which is earlier in the zib-order: the skind or the floane? Reply with only one entity name. No explanation.  **→ floane**
- `rank` — Counting the first as position 1, what is the sloarl's position in the order? Reply with only the number. No explanation.  **→ 2**
- `rank` — Counting the first as position 1, what is the throamb's position in the order? Reply with only the number. No explanation.  **→ 3**
- `betweenness` — Using only the relations stated, which of these is between the other two in the order (counting the first as position 1): the pronnel, the floane, or the throamb? Reply with only one entity name. No explanation.  **→ pronnel**
- `betweenness` — Using only the relations stated, which of these is between the other two in the order (counting the first as position 1): the sloarl, the gleendrel, or the skind? Reply with only one entity name. No explanation.  **→ skind**
- `successor` — Using only the relations stated, which entity is immediately after the flend in the order (the next position toward the last, counting the first as position 1)? Reply with only the entity name. No explanation.  **→ pronnel**
- `successor` — Using only the relations stated, which entity is immediately after the gleendrel in the order (the next position toward the last, counting the first as position 1)? Reply with only the entity name. No explanation.  **→ trostrick**
- `predecessor` — Using only the relations stated, which entity is immediately before the pronnel in the order (the next position toward the first, counting the first as position 1)? Reply with only the entity name. No explanation.  **→ flend**
- `predecessor` — Using only the relations stated, which entity is immediately before the sloarl in the order (the next position toward the first, counting the first as position 1)? Reply with only the entity name. No explanation.  **→ greemth**
- `count_between` — Using only the relations stated, how many entities are strictly between the floane and the pronnel in the order? Reply with only the number. No explanation.  **→ 0**
- `count_between` — Using only the relations stated, how many entities are strictly between the skind and the sloarl in the order? Reply with only the number. No explanation.  **→ 4**
- `comparative_distance` — Using only the relations stated, which is closer to the pronnel in the order: the throamb or the floane? Reply with only one entity name. No explanation.  **→ floane**
- `comparative_distance` — Using only the relations stated, which is closer to the pronnel in the order: the gleendrel or the skind? Reply with only one entity name. No explanation.  **→ skind**
- `extremes` — Using only the relations stated, which entity is the first of all — the one that comes before every other entity in the order? Reply with only the entity name. No explanation.  **→ greemth**
- `extremes` — Using only the relations stated, which entity is the last of all — the one that comes after every other entity in the order? Reply with only the entity name. No explanation.  **→ trostrick**

---

## Total order · S1 size · N=9 · easy · FORWARD (ceiling/control)

*Same content as the shuffle twin; forward = cards sorted by rank (position leaks order).*

**latent order (rank 1..N):** 1:thrindrel < 2:drend < 3:groaft < 4:smeene < 5:spiemb < 6:skine < 7:speerl < 8:crulb < 9:glizzle

**prompt:**
```
The thrindrel is smaller than the drend.
The groaft is larger than the thrindrel.
The thrindrel is smaller than the smeene.
The glizzle is larger than the thrindrel.
The drend is smaller than the groaft.
The smeene is larger than the drend.
The drend is smaller than the crulb.
The groaft is smaller than the smeene.
The spiemb is larger than the groaft.
The smeene is smaller than the spiemb.
The skine is larger than the spiemb.
The spiemb is smaller than the speerl.
The speerl is larger than the skine.
The crulb is larger than the skine.
The skine is smaller than the glizzle.
The speerl is smaller than the crulb.
The glizzle is larger than the speerl.
The crulb is smaller than the glizzle.

Entities: the glizzle, the drend, the speerl, the thrindrel, the smeene, the spiemb, the skine, the crulb, the groaft.
```

**questions (family → key):**

---

## Total order · S1 heat (cooler/hotter) · N=9 · easy · shuffle

*Extra S1 relation family (semantics-gradient robustness).*

**latent order (rank 1..N):** 1:driemp < 2:flune < 3:speeft < 4:spax < 5:quept < 6:grerl < 7:thronnel < 8:quiemb < 9:fleept

**prompt:**
```
The thronnel is cooler than the quiemb.
The grerl is cooler than the fleept.
The flune is cooler than the quept.
The flune is cooler than the speeft.
The driemp is cooler than the flune.
The quept is cooler than the thronnel.
The fleept is hotter than the driemp.
The grerl is hotter than the quept.
The driemp is cooler than the spax.
The fleept is hotter than the thronnel.
The spax is hotter than the flune.
The speeft is cooler than the quiemb.
The spax is hotter than the speeft.
The thronnel is hotter than the grerl.
The quiemb is hotter than the grerl.
The quiemb is cooler than the fleept.
The speeft is hotter than the driemp.
The quept is hotter than the spax.

Entities: the quiemb, the speeft, the flune, the thronnel, the spax, the fleept, the driemp, the grerl, the quept.
```

**questions (family → key):**
- `reconstruction` — Using only the relations stated above, list all entities from the coolest to the hottest (this order may differ from the order the lines appear in). Reply with one entity name per line, nothing else.  **→ ['driemp', 'flune', 'speeft', 'spax', 'quept', 'grerl', 'thronnel', 'quiemb', 'fleept']**
- `reconstruction` — List all entities in the order they first appear in the text above, top to bottom. Reply with one entity name per line, nothing else.  **→ MENTION_ORDER**
- `pairwise` — By the relations above, which is cooler: the grerl or the thronnel? Reply with only one entity name. No explanation.  **→ grerl**
- `pairwise` — By the relations above, which is cooler: the thronnel or the grerl? Reply with only one entity name. No explanation.  **→ grerl**
- `rank` — Counting the coolest as position 1, what is the driemp's position in the order? Reply with only the number. No explanation.  **→ 1**
- `rank` — Counting the coolest as position 1, what is the flune's position in the order? Reply with only the number. No explanation.  **→ 2**
- `betweenness` — Using only the relations stated, which of these is between the other two in the order (counting the coolest as position 1): the spax, the grerl, or the speeft? Reply with only one entity name. No explanation.  **→ spax**
- `betweenness` — Using only the relations stated, which of these is between the other two in the order (counting the coolest as position 1): the quept, the fleept, or the driemp? Reply with only one entity name. No explanation.  **→ quept**
- `successor` — Using only the relations stated, which entity is immediately after the grerl in the order (the next position toward the hottest, counting the coolest as position 1)? Reply with only the entity name. No explanation.  **→ thronnel**
- `successor` — Using only the relations stated, which entity is immediately after the quiemb in the order (the next position toward the hottest, counting the coolest as position 1)? Reply with only the entity name. No explanation.  **→ fleept**
- `predecessor` — Using only the relations stated, which entity is immediately before the spax in the order (the next position toward the coolest, counting the coolest as position 1)? Reply with only the entity name. No explanation.  **→ speeft**
- `predecessor` — Using only the relations stated, which entity is immediately before the speeft in the order (the next position toward the coolest, counting the coolest as position 1)? Reply with only the entity name. No explanation.  **→ flune**
- `count_between` — Using only the relations stated, how many entities are strictly between the grerl and the thronnel in the order? Reply with only the number. No explanation.  **→ 0**
- `count_between` — Using only the relations stated, how many entities are strictly between the fleept and the spax in the order? Reply with only the number. No explanation.  **→ 4**
- `comparative_distance` — Using only the relations stated, which is closer to the spax in the order: the quept or the grerl? Reply with only one entity name. No explanation.  **→ quept**
- `comparative_distance` — Using only the relations stated, which is closer to the flune in the order: the thronnel or the quiemb? Reply with only one entity name. No explanation.  **→ thronnel**
- `extremes` — Using only the relations stated, which entity is the coolest of all — the one that comes before every other entity in the order? Reply with only the entity name. No explanation.  **→ driemp**
- `extremes` — Using only the relations stated, which entity is the hottest of all — the one that comes after every other entity in the order? Reply with only the entity name. No explanation.  **→ fleept**

---

## Total order · S1 size · N=24 · hard · shuffle (length stress)

*Large-N stress cell (SSM state-bottleneck test).*

**latent order (rank 1..N):** 1:fleex < 2:glizzle < 3:breesk < 4:floarl < 5:clanch < 6:troaft < 7:quax < 8:brone < 9:drend < 10:slumb < 11:blunnel < 12:drierl < 13:pript < 14:drunnel < 15:crarvic < 16:speesk < 17:grumth < 18:smane < 19:pronnel < 20:glunnel < 21:froane < 22:thrieft < 23:gliench < 24:sniern

**prompt:**
```
The grumth is smaller than the smane.
The pronnel is larger than the clanch.
The crarvic is smaller than the speesk.
The pript is smaller than the drunnel.
The glizzle is smaller than the crarvic.
The smane is smaller than the pronnel.
The crarvic is larger than the quax.
The thrieft is larger than the fleex.
The froane is larger than the breesk.
The fleex is smaller than the glizzle.
… (50 lines total)
```

**questions (family → key):**

---

## Coherence-null twin · S1 size · N=9

*An INVALID cycle is injected (no valid total order). Must decode at chance. Distinct from the cyclic ring below (which is a VALID cycle).*

**latent order (rank 1..N):** 1:promth < 2:brend < 3:blask < 4:slastrick < 5:fliemb < 6:stulb < 7:smeene < 8:prarvic < 9:bleeft

**prompt:**
```
The prarvic is larger than the brend.
The stulb is larger than the fliemb.
The smeene is larger than the blask.
The smeene is larger than the promth.
The blask is smaller than the fliemb.
The fliemb is smaller than the bleeft.
The promth is smaller than the prarvic.
The stulb is larger than the brend.
The blask is smaller than the slastrick.
The promth is smaller than the slastrick.
The prarvic is larger than the smeene.
The prarvic is smaller than the bleeft.
… (20 lines total)
```

**questions (family → key):**

---

## Partial order · 2 chains [5,4] · S1 size · shuffle

*Cross-chain pairs are INCOMPARABLE (order_query mixes comparable + 'undetermined').*

**chains:** chain0=[quoarvic,trostrick,dristrick,dreench,glerl]; chain1=[snendrel,frustrick,snorl,skirvic]

**prompt:**
```
The glerl is larger than the quoarvic.
The dreench is smaller than the glerl.
The quoarvic is smaller than the dreench.
The frustrick is smaller than the snorl.
The dreench is larger than the trostrick.
The skirvic is larger than the snendrel.
The dristrick is smaller than the dreench.
The trostrick is smaller than the dristrick.
The glerl is larger than the dristrick.
The snendrel is smaller than the frustrick.
The quoarvic is smaller than the trostrick.
The trostrick is smaller than the glerl.
The dristrick is larger than the quoarvic.
The snorl is smaller than the skirvic.

Entities: the skirvic, the dreench, the trostrick, the dristrick, the quoarvic, the frustrick, the glerl, the snorl, the snendrel.
```

**questions (family → key):**
- `order_query` — Using only the relations stated, is it determined which is smaller, the skirvic or the snorl? Answer with that entity's name if determined, otherwise answer 'undetermined'.  **→ snorl**
- `order_query` — Using only the relations stated, is it determined which is smaller, the snorl or the skirvic? Answer with that entity's name if determined, otherwise answer 'undetermined'.  **→ snorl**

---

## 2-D grid · size × loud · N=9 · shuffle

*Two INDEPENDENT global orders over the same entities; per-axis pairwise.*

**coords (entity: x,y):** quax:1,3, slind:2,6, prennel:3,9, drumb:4,5, blumb:5,8, plestrick:6,7, sloanch:7,4, prumth:8,1 …

**prompt:**
```
The slind is louder than the drumb.
The plestrick is louder than the slind.
The quax is smaller than the slind.
The tronnel is quieter than the prennel.
The prennel is smaller than the drumb.
The plestrick is louder than the quax.
The blumb is smaller than the sloanch.
The plestrick is smaller than the sloanch.
The prumth is quieter than the slind.
The prumth is quieter than the tronnel.
The slind is smaller than the prennel.
The quax is quieter than the blumb.
The sloanch is smaller than the prumth.
The prumth is smaller than the tronnel.
The tronnel is larger than the quax.
The drumb is louder than the prumth.
The quax is quieter than the sloanch.
The plestrick is larger than the blumb.
The blumb is larger than the drumb.
The tronnel is larger than the plestrick.
The tronnel is quieter than the quax.
The slind is louder than the tronnel.
The blumb is louder than the plestrick.
The drumb is larger than the quax.
The drumb is quieter than the prennel.
The quax is smaller than the blumb.
The prennel is louder than the sloanch.
The sloanch is larger than the slind.
The drumb is smaller than the plestrick.
The blumb is louder than the prumth.
The sloanch is quieter than the drumb.
The prumth is larger than the prennel.
The prennel is louder than the blumb.
The slind is smaller than the prumth.
The sloanch is quieter than the plestrick.
The prennel is smaller than the tronnel.

Entities: the sloanch, the drumb, the plestrick, the blumb, the prumth, the prennel, the quax, the tronnel, the slind.
```

**questions (family → key):**
- `pairwise` — By the relations above, which is smaller: the plestrick or the prennel? Reply with only one entity name. No explanation.  **→ prennel**
- `pairwise` — By the relations above, which is smaller: the prennel or the plestrick? Reply with only one entity name. No explanation.  **→ prennel**

---

## Cyclic ring · S1 size · N=12 · shuffle (nonlinear litmus)

*A VALID single cycle (positions wrap). No endpoints; first-named ⟂ position (Eulerian). Predicts a CIRCULAR manifold.*

**cyclic order (pos:entity):** 0:spustrick → 1:smomth → 2:skirn → 3:slirn → 4:quinch → 5:grarl → 6:cliezzle → 7:stumth → 8:drestrick → 9:creemp → 10:throrvic → 11:standrel → (wrap)

**prompt:**
```
In this puzzle the entities are arranged clockwise around a circle and the positions wrap around, so every entity has both a clockwise-next and a clockwise-previous entity. 'The A is k place(s) before the B' means the B is k steps clockwise from the A; equivalently, 'The B is k place(s) after the A' means the same thing.

The stumth is 2 places after the grarl.
The slirn is 2 places after the smomth.
The drestrick is 1 place before the creemp.
The spustrick is 2 places after the throrvic.
The creemp is 1 place before the throrvic.
The grarl is 2 places after the slirn.
The skirn is 1 place before the slirn.
The standrel is 2 places after the creemp.
The quinch is 1 place before the grarl.
The grarl is 1 place before the cliezzle.
The smomth is 1 place before the skirn.
The throrvic is 2 places after the drestrick.
The skirn is 2 places after the spustrick.
The cliezzle is 1 place before the stumth.
The smomth is 2 places after the standrel.
The creemp is 2 places after the stumth.
The spustrick is 1 place before the smomth.
The stumth is 1 place before the drestrick.
The cliezzle is 2 places after the quinch.
The standrel is 1 place before the spustrick.
The quinch is 2 places after the skirn.
The slirn is 1 place before the quinch.
The drestrick is 2 places after the cliezzle.
The throrvic is 1 place before the standrel.

Entities: the grarl, the slirn, the quinch, the throrvic, the cliezzle, the creemp, the drestrick, the stumth, the smomth, the standrel, the skirn, the spustrick.
```

**questions (family → key):**
- `cyclic_successor` — Going clockwise around the circle, which entity comes immediately after the standrel? Reply with only the entity name. No explanation.  **→ spustrick**
- `cyclic_successor` — Going clockwise around the circle, which entity comes immediately after the skirn? Reply with only the entity name. No explanation.  **→ slirn**
- `cyclic_predecessor` — Going clockwise around the circle, which entity comes immediately before the stumth? Reply with only the entity name. No explanation.  **→ cliezzle**
- `cyclic_predecessor` — Going clockwise around the circle, which entity comes immediately before the skirn? Reply with only the entity name. No explanation.  **→ smomth**
- `cyclic_distance` — Going clockwise from the smomth, how many steps until you reach the standrel? Reply with only the number. No explanation.  **→ 10**
- `cyclic_distance` — Going clockwise from the spustrick, how many steps until you reach the creemp? Reply with only the number. No explanation.  **→ 9**
- `cyclic_order` — Going clockwise from the drestrick, which do you reach first: the creemp or the slirn? Reply with only one entity name. No explanation.  **→ creemp**
- `cyclic_order` — Going clockwise from the throrvic, which do you reach first: the stumth or the cliezzle? Reply with only one entity name. No explanation.  **→ cliezzle**

---

## Cyclic ring · S0 zib · N=9 · shuffle

**cyclic order (pos:entity):** 0:glalb → 1:quiemb → 2:quump → 3:smeend → 4:fleept → 5:glusk → 6:snomth → 7:clonnel → 8:sleex → (wrap)

**prompt:**
```
In this puzzle the entities are arranged clockwise around a circle and the positions wrap around, so every entity has both a clockwise-next and a clockwise-previous entity. 'The A is k place(s) before the B' means the B is k steps clockwise from the A; equivalently, 'The B is k place(s) after the A' means the same thing.

The glalb is 2 places after the clonnel.
The quiemb is 1 place before the quump.
The glusk is 1 place before the snomth.
The clonnel is 1 place before the sleex.
The glalb is 1 place before the quiemb.
The fleept is 1 place before the glusk.
The smeend is 2 places after the quiemb.
The snomth is 2 places after the fleept.
The smeend is 1 place before the fleept.
The sleex is 2 places after the snomth.
The fleept is 2 places after the quump.
The quump is 1 place before the smeend.
The quiemb is 2 places after the sleex.
The glusk is 2 places after the smeend.
The snomth is 1 place before the clonnel.
The clonnel is 2 places after the glusk.
The quump is 2 places after the glalb.
The sleex is 1 place before the glalb.

Entities: the quiemb, the glusk, the quump, the glalb, the fleept, the smeend, the sleex, the snomth, the clonnel.
```

**questions (family → key):**
- `cyclic_successor` — Going clockwise around the circle, which entity comes immediately after the fleept? Reply with only the entity name. No explanation.  **→ glusk**
- `cyclic_successor` — Going clockwise around the circle, which entity comes immediately after the quiemb? Reply with only the entity name. No explanation.  **→ quump**

