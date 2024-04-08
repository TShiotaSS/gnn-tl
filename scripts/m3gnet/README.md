# Modified file

## _atom_ref.py
- class : AtomRef<br>
- def : call<br> 
Adding each atom ref energy in the return.

## _readout.py
class : ReduceReadOut<br>
def : call<br>
Adding atomic number infomation in the return.

## _base.py
python decorator
```
@tf.function(experimental_relax_shapes=True)
```

## _m3gnet.py
- class : M3GNet <br>
- def : call <br>
Adding each atom energy, vector information, then outputtin as csv.
