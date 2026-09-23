"""
Script to parse JARVIS-EFG4.json
Requires javris-tools installation.
Follow installation instructions on https://github.com/usnistgov/jarvis
"""

from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
import pandas as pd
import numpy as np

data = loadjson("JARVIS-EFG4.json")
df = pd.DataFrame(data).set_index("jid")

search_id = "JVASP-5"  # example JARVIS-ID
result = df.ix[search_id]
atoms = Atoms.from_dict(result["atoms"])
spg = Spacegroup3D(atoms)
wycs = spg._dataset["wyckoffs"]
spg_numb = str(spg.space_group_number)
spg_symb = str(spg.space_group_symbol)
pg_symb = str(spg.point_group_symbol)
crystal_system = str(spg.crystal_system)
natoms = atoms.num_atoms
formula = str(atoms.composition.reduced_formula)
combinations = []
print("JARVIS-ID", search_id)
print("Formula", formula)
print("Spacegroup related info:", spg_numb, spg_symb, pg_symb, crystal_system)
for k in range(natoms):
    comb = str(atoms.elements[k]) + "," + str(wycs[k])
    if comb not in combinations:
        combinations.append(comb)
        print(
            "Diag_Vxx, Diag_Vyy, Diag_Vzz, eta =",
            atoms.elements[k],
            wycs[k],
            np.array(result["efg_diag_tensor"][k], dtype="float") / 10.0,
        )
        print(
            "VASP Vxx...Vzz",
            np.array(result["efg_raw_tensor"][k], dtype="float") / 10.0,
        )
