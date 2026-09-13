# BCS — примеры промптов (то, что подаётся в модель)

## s0_zib — абстрактная nonce-связь «zibs»

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The drund zibs the glundrel.
The queench zibs the flune.
The spund zibs the drund.
The flune zibs the promth.
The glundrel is zibbed by the flune.
The promth is zibbed by the queench.
The spund zibs the promth.
The flune is zibbed by the triemth.
The breemb zibs the cleene.
The drund zibs the queench.
The queench zibs the breemb.
The cleene zibs the glundrel.
The breemb is zibbed by the drund.
The glundrel is zibbed by the triemth.
The cleene is zibbed by the spund.
The triemth zibs the cleene.
The triemth zibs the spund.
The promth zibs the breemb.

Entities: the queench, the spund, the glundrel, the flune, the breemb, the cleene, the drund, the promth, the triemth.
```

## s0_quomp — абстрактная nonce-связь «quomps»

```
In this puzzle, 'quomps' is a transitive relation: if X quomps Y and Y quomps Z, then X quomps Z. 'X quomps Y' means X comes before Y in the quomp-order.

The thrinnel quomps the stemp.
The plene quomps the plurn.
The drieft is quomped by the sliex.
The brilb quomps the drieft.
The sniept is quomped by the drieft.
The plurn quomps the sliex.
The slulb quomps the thrinnel.
The sliex is quomped by the brilb.
The brilb quomps the plene.
The sniept is quomped by the brilb.
The slulb quomps the stemp.
The sliex is quomped by the thrinnel.
The drieft is quomped by the slulb.
The stemp is quomped by the sniept.
The plene quomps the sniept.
The plurn quomps the slulb.
The stemp is quomped by the plene.
The thrinnel is quomped by the plurn.

Entities: the thrinnel, the plene, the sliex, the drieft, the plurn, the stemp, the slulb, the sniept, the brilb.
```

## s1_size — семантика: larger/smaller

```
The drunnel is smaller than the spund.
The spund is larger than the spiemb.
The drunnel is smaller than the spiemb.
The smoarl is smaller than the gloask.
The smorl is smaller than the slind.
The smoarl is smaller than the triern.
The gloask is smaller than the slind.
The smorl is smaller than the smoarl.
The triern is smaller than the grastrick.
The grastrick is larger than the drunnel.
The slind is larger than the smoarl.
The slind is smaller than the grastrick.
The spiemb is smaller than the gloask.
The triern is smaller than the drunnel.
The spund is larger than the smorl.
The gloask is larger than the triern.
The spiemb is larger than the smorl.
The grastrick is smaller than the spund.

Entities: the smorl, the spund, the drunnel, the gloask, the slind, the triern, the smoarl, the grastrick, the spiemb.
```

## s1_loud — семантика: louder/quieter

```
The troft is quieter than the glinnel.
The preemth is quieter than the gliept.
The smundrel is quieter than the gliept.
The slorl is louder than the threzzle.
The glinnel is louder than the smundrel.
The groax is quieter than the troft.
The gliept is louder than the troft.
The threzzle is quieter than the spoane.
The threzzle is quieter than the preemth.
The spoane is louder than the glinnel.
The smundrel is louder than the threzzle.
The gliept is louder than the slorl.
The preemth is quieter than the groax.
The troft is quieter than the smundrel.
The spoane is louder than the preemth.
The groax is quieter than the slorl.
The glinnel is louder than the groax.
The slorl is louder than the spoane.

Entities: the spoane, the threzzle, the groax, the gliept, the troft, the slorl, the preemth, the glinnel, the smundrel.
```

## s1_heat — семантика: hotter/colder

```
The smipt is cooler than the snerl.
The smorl is cooler than the glilb.
The prennel is cooler than the snerl.
The quiemb is hotter than the prennel.
The smipt is hotter than the smorl.
The prennel is cooler than the stiemp.
The broastrick is cooler than the thrinnel.
The snerl is hotter than the thrinnel.
The thrinnel is cooler than the smipt.
The quiemb is hotter than the broastrick.
The stiemp is cooler than the broastrick.
The stiemp is cooler than the quiemb.
The thrinnel is cooler than the smorl.
The glilb is hotter than the stiemp.
The smorl is hotter than the prennel.
The glilb is cooler than the smipt.
The broastrick is cooler than the glilb.
The snerl is cooler than the quiemb.

Entities: the smorl, the thrinnel, the prennel, the quiemb, the smipt, the glilb, the broastrick, the stiemp, the snerl.
```

## coherence-twin (s0_zib, инъектирован цикл → валидного порядка нет)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The glundrel is zibbed by the drund.
The flune is zibbed by the queench.
The drund is zibbed by the spund.
The promth is zibbed by the flune.
The flune zibs the glundrel.
The promth is zibbed by the queench.
The promth zibs the spund.
The triemth zibs the flune.
The breemb zibs the cleene.
The drund zibs the queench.
The breemb is zibbed by the queench.
The glundrel is zibbed by the cleene.
The breemb is zibbed by the drund.
The glundrel is zibbed by the triemth.
The cleene is zibbed by the spund.
The triemth zibs the cleene.
The triemth zibs the spund.
The breemb is zibbed by the promth.

Entities: the queench, the spund, the glundrel, the flune, the breemb, the cleene, the drund, the promth, the triemth.
```

## condition = forward (карточки по возрастанию ранга; s1_size)

```
The broax is smaller than the tronnel.
The treex is larger than the broax.
The broax is smaller than the quoamb.
The croand is larger than the broax.
The tronnel is smaller than the floamth.
The quoamb is larger than the tronnel.
The tronnel is smaller than the croand.
The floamth is smaller than the bleemp.
The floamth is smaller than the triex.
The gleeft is larger than the floamth.
The bleemp is smaller than the treex.
The bleemp is smaller than the gleeft.
The croand is larger than the bleemp.
The treex is smaller than the quoamb.
The triex is larger than the treex.
The quoamb is smaller than the triex.
The triex is smaller than the gleeft.
The gleeft is smaller than the croand.

Entities: the triex, the gleeft, the tronnel, the floamth, the bleemp, the broax, the quoamb, the treex, the croand.
```

## condition = shuffle (те же связи, перемешаны; s1_size)

```
The triex is smaller than the gleeft.
The treex is larger than the broax.
The broax is smaller than the quoamb.
The tronnel is smaller than the floamth.
The quoamb is smaller than the triex.
The triex is larger than the treex.
The bleemp is smaller than the treex.
The tronnel is smaller than the croand.
The gleeft is smaller than the croand.
The quoamb is larger than the tronnel.
The gleeft is larger than the floamth.
The croand is larger than the broax.
The floamth is smaller than the bleemp.
The broax is smaller than the tronnel.
The treex is smaller than the quoamb.
The croand is larger than the bleemp.
The floamth is smaller than the triex.
The bleemp is smaller than the gleeft.

Entities: the tronnel, the quoamb, the croand, the gleeft, the broax, the bleemp, the treex, the triex, the floamth.
```

## difficulty = easy (локальные цепочки; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The stannel is zibbed by the gliept.
The thrarl is zibbed by the spoane.
The plene is zibbed by the thrundrel.
The clept zibs the slask.
The thrundrel zibs the slask.
The gliept zibs the plene.
The thrarl zibs the clept.
The sliex zibs the thrundrel.
The plene zibs the stannel.
The thrundrel zibs the gliept.
The clept is zibbed by the stannel.
The spoane is zibbed by the plene.
The sliex zibs the thrarl.
The slask is zibbed by the thrarl.
The slask is zibbed by the sliex.
The spoane zibs the clept.
The gliept is zibbed by the sliex.
The stannel zibs the spoane.

Entities: the stannel, the slask, the sliex, the thrarl, the thrundrel, the gliept, the spoane, the clept, the plene.
```

## difficulty = hard (дальние рёбра, нужна глобальная интеграция; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The thrarl is zibbed by the plene.
The plene zibs the slask.
The spoane is zibbed by the gliept.
The clept is zibbed by the sliex.
The spoane zibs the thrarl.
The stannel is zibbed by the thrundrel.
The sliex zibs the spoane.
The thrundrel zibs the gliept.
The stannel zibs the spoane.
The plene zibs the stannel.
The gliept zibs the thrarl.
The clept is zibbed by the stannel.
The gliept zibs the plene.
The slask is zibbed by the clept.
The thrundrel zibs the slask.
The slask is zibbed by the sliex.
The thrarl zibs the clept.
The sliex zibs the thrundrel.

Entities: the spoane, the stannel, the sliex, the clept, the thrarl, the plene, the gliept, the slask, the thrundrel.
```

## D2 declared-list (полный порядок ЗАЯВЛЕН, не выводится; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The complete order, from earliest to latest, is stated below.

Order: the snandrel, then the troaft, then the cloask, then the prarvic, then the snomp, then the groaft, then the drorvic, then the frervic, then the grurl.

Entities: the frervic, the troaft, the groaft, the prarvic, the cloask, the drorvic, the snomp, the snandrel, the grurl.
```

## D3 declared-adjacency (только соседние пары; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The frervic zibs the grurl.
The cloask zibs the prarvic.
The snandrel zibs the troaft.
The prarvic zibs the snomp.
The snomp zibs the groaft.
The troaft zibs the cloask.
The groaft zibs the drorvic.
The drorvic zibs the frervic.

Entities: the frervic, the prarvic, the drorvic, the snomp, the snandrel, the cloask, the troaft, the groaft, the grurl.
```

## D4 derived + summary (выводимые связи + итоговая строка порядка; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The frervic is zibbed by the cloask.
The snomp is zibbed by the snandrel.
The groaft zibs the drorvic.
The troaft zibs the cloask.
The prarvic zibs the drorvic.
The grurl is zibbed by the snomp.
The grurl is zibbed by the snandrel.
The cloask zibs the grurl.
The cloask zibs the prarvic.
The frervic zibs the grurl.
The snandrel zibs the troaft.
The snomp zibs the groaft.
The troaft zibs the frervic.
The drorvic is zibbed by the troaft.
The drorvic zibs the frervic.
The prarvic zibs the snomp.
The snandrel zibs the groaft.
The groaft is zibbed by the prarvic.

The complete order, from earliest to latest, is: the snandrel, then the troaft, then the cloask, then the prarvic, then the snomp, then the groaft, then the drorvic, then the frervic, then the grurl.

Entities: the prarvic, the frervic, the grurl, the cloask, the snandrel, the snomp, the troaft, the drorvic, the groaft.
```

## hop-dial reach=2 (связи gap 1 и 2; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The quirn zibs the quoamb.
The snorl zibs the bleelb.
The plorl is zibbed by the plipt.
The fleept zibs the spiene.
The bleelb is zibbed by the plorl.
The drund is zibbed by the fleept.
The queept is zibbed by the spiene.
The spiene zibs the drund.
The plipt is zibbed by the drund.
The snorl zibs the skiezzle.
The plorl zibs the quoamb.
The drund zibs the queept.
The quirn zibs the snorl.
The skiezzle zibs the fleept.
The fleept is zibbed by the snorl.
The quoamb is zibbed by the bloanch.
The quoamb zibs the bleelb.
The plipt zibs the bloanch.
The bleelb is zibbed by the quirn.
The bloanch zibs the plorl.
The spiene is zibbed by the skiezzle.
The queept zibs the plipt.
The bloanch is zibbed by the queept.
The skiezzle is zibbed by the quirn.

Entities: the spiene, the drund, the plorl, the plipt, the quoamb, the snorl, the quirn, the fleept, the bleelb, the queept, the skiezzle, the bloanch.
```

## hop-dial reach=4 (связи gap 1 и 4 — глубже вывод; s0_zib)

```
In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.

The queept is zibbed by the skiezzle.
The plipt zibs the bloanch.
The quirn zibs the bloanch.
The plipt is zibbed by the fleept.
The bleelb is zibbed by the plipt.
The snorl zibs the skiezzle.
The fleept zibs the bleelb.
The quoamb zibs the bleelb.
The snorl zibs the plorl.
The queept zibs the plipt.
The quirn zibs the snorl.
The spiene is zibbed by the quirn.
The drund is zibbed by the snorl.
The spiene zibs the drund.
The bloanch zibs the plorl.
The bleelb is zibbed by the quirn.
The skiezzle zibs the quoamb.
The quoamb is zibbed by the queept.
The skiezzle zibs the fleept.
The bloanch is zibbed by the spiene.
The fleept zibs the spiene.
The plorl zibs the quoamb.
The plorl is zibbed by the drund.
The drund zibs the queept.

Entities: the plipt, the snorl, the spiene, the fleept, the bleelb, the plorl, the skiezzle, the queept, the drund, the bloanch, the quoamb, the quirn.
```

## structure = cyclic (кольцо; s0_zib)

```
In this puzzle the entities are arranged clockwise around a circle and the positions wrap around, so every entity has both a clockwise-next and a clockwise-previous entity. 'The A is k place(s) before the B' means the B is k steps clockwise from the A; equivalently, 'The B is k place(s) after the A' means the same thing.

The troarl is 1 place before the frustrick.
The crurvic is 2 places after the triex.
The troarl is 2 places after the brerl.
The treennel is 2 places after the troarl.
The grarl is 2 places after the frustrick.
The grarl is 1 place before the drine.
The drine is 1 place before the briench.
The briench is 1 place before the triex.
The triex is 2 places after the drine.
The crurvic is 1 place before the troarl.
The treennel is 1 place before the grarl.
The briench is 2 places after the grarl.
The brerl is 2 places after the briench.
The triex is 1 place before the brerl.
The drine is 2 places after the treennel.
The brerl is 1 place before the crurvic.
The frustrick is 1 place before the treennel.
The frustrick is 2 places after the crurvic.

Entities: the brerl, the triex, the crurvic, the troarl, the briench, the treennel, the frustrick, the grarl, the drine.
```

## structure = grid2d (2D-решётка size×loud)

```
The glizzle is louder than the spunch.
The cleelb is louder than the drirvic.
The plapt is smaller than the spunch.
The plapt is quieter than the steerl.
The thrisk is louder than the smomth.
The spunch is larger than the thrisk.
The croarvic is quieter than the glizzle.
The plapt is louder than the glizzle.
The drirvic is smaller than the cleelb.
The steerl is louder than the croarvic.
The steerl is smaller than the glizzle.
The smomth is larger than the steerl.
The croarvic is louder than the cleelb.
The cleelb is quieter than the steerl.
The cleelb is larger than the steerl.
The glizzle is larger than the drirvic.
The croarvic is larger than the cleelb.
The spunch is quieter than the plapt.
The smomth is quieter than the spunch.
The croarvic is smaller than the smomth.
The thrisk is smaller than the croarvic.
The steerl is louder than the thrisk.
The spunch is quieter than the drirvic.
The thrisk is smaller than the plapt.
The smomth is smaller than the spunch.
The spunch is smaller than the glizzle.
The glizzle is larger than the croarvic.
The plapt is larger than the drirvic.
The glizzle is quieter than the thrisk.
The thrisk is quieter than the plapt.
The steerl is larger than the plapt.
The drirvic is louder than the smomth.
The drirvic is smaller than the thrisk.
The cleelb is smaller than the smomth.
The smomth is quieter than the cleelb.
The drirvic is quieter than the croarvic.

Entities: the glizzle, the plapt, the steerl, the cleelb, the drirvic, the thrisk, the smomth, the spunch, the croarvic.
```

## structure = partial_order (несколько несвязанных цепочек; s1_size)

```
The quennel is smaller than the plipt.
The drine is smaller than the smuft.
The smuft is smaller than the glendrel.
The glendrel is smaller than the pranch.
The plipt is smaller than the cloalb.
The cloalb is larger than the quirn.
The pranch is larger than the drine.
The quirn is smaller than the quennel.

Entities: the glendrel, the pranch, the quennel, the drine, the cloalb, the plipt, the smuft, the quirn.
```

## Вопросы батареи (тоже подаются в модель, к каждому стимулу)

```
[reconstruction] Using only the relations stated above, list all entities from the smallest to the largest (this order may differ from the order the lines appear in). Reply with one entity name per line, nothing else.

[pairwise] By the relations above, which is smaller: the spiemb or the gloask? Reply with only one entity name. No explanation.

[rank] Counting the smallest as position 1, what is the smoarl's position in the order? Reply with only the number. No explanation.

[betweenness] Using only the relations stated, which of these is between the other two in the order (counting the smallest as position 1): the gloask, the triern, or the slind? Reply with only one entity name. No explanation.

[successor] Using only the relations stated, which entity is immediately after the triern in the order (the next position toward the largest, counting the smallest as position 1)? Reply with only the entity name. No explanation.

[predecessor] Using only the relations stated, which entity is immediately before the gloask in the order (the next position toward the smallest, counting the smallest as position 1)? Reply with only the entity name. No explanation.

[count_between] Using only the relations stated, how many entities are strictly between the slind and the gloask in the order? Reply with only the number. No explanation.

[comparative_distance] Using only the relations stated, which is closer to the drunnel in the order: the triern or the gloask? Reply with only one entity name. No explanation.

[extremes] Using only the relations stated, which entity is the smallest of all — the one that comes before every other entity in the order? Reply with only the entity name. No explanation.

```
