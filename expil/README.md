
#### Prepare Data
``` 
python -m nesy_pi.collect_data_getout -m getout --device 10
```

#### Predicate Invention and Rule Reasoning
``` 
python -m nesy_pi.aaa_main -m getout --with_pi --show_process --device 0
python -m nesy_pi.aaa_main -m loot --with_pi --show_process --device 0 
python -m nesy_pi.aaa_main -m threefish --with_pi --show_process --device 0
```
#### Weight Learning
``` 
python -m nesy_pi.train_nudge -m getout -alg logic -env getout -r getout_pi --with_pi -s 0  --device 0
python -m nesy_pi.train_nudge -m loot -alg logic -env loot -r loot_pi --with_pi -s 0 --device 0
python -m nesy_pi.train_nudge -m threefish -alg logic -env threefish -r threefish_pi --with_pi -s 0  --device 0
``` 