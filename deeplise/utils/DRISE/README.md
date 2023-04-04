
# Dependencies
- install freesasa with instructions http://freesasa.github.io/
  - ensure that libxml2-devel was installed
  - ensure that libjson-c-dev is installed
- CUDA toolkit

# Compilation
- Simply call make inside the DRISE directory
- Compile for debug with make argument: `make DEBUG=ON`

# Execution
### Parameters do not need to be in order, the flags just need to be specified
  - Storage Connector Type = `-c`
    - options = file, local, or remote
    - local and remote currently not functional
  - Query Molecule Type = `-m`
    - options = protein, ligand, dna, rna, carbohydrate
    - currently protein is the only supported target
      - this feature is in the works
  - Input = `-i`
    - path to pdb data which could be:
      - single pdb file
      - directory of pdb files
      - single ISE JSON
      - directory of ISE JSONs
      - pdbID (not currently available)
      - pdbID.txt (not currently available)
  - Process/ISE Configuration = `-p`
    - options = atomtype, surface, train, or bsp
  - Atom Typing scheme = `-a`
    - value should be location to typing csv (default and example live in data/atomtypes)
### Bash scripts
  - ./util/parameterized_test.sh
  - first parameter is location of input
  - second parameter is query type



# Output (all output to ./data/)
### Debug outputs (compiled with DEBUG=ON)
  - surface checker in pdb format, beta factor = surface id
  - atom type checker in custom format
### Complex JSONs
  - these are generated from methods in io_uti.cpp
  - have atoms organized into molecules and residues
### Sites
  - site checkers in pdb format similar to surface checker, but affinity = beta factor


# TODO
- make biofile io class
  - should include separateMolecules(), parsePDB, parseMol2, etc.
- ensure multiple query types are allowed
- support non-protein target type
- allow non-standard residue typing
- type non-protein atoms
- type nonstandard residues
- type molecules with multiple residue types
  - may be able to consider dominating molecule types (i.e. protein with cofactor is still protein)
- fix printouts
- ensure when meta triangles are being updated in particleGraph.updateTrianglesFromSpheres() that atom types are adjusted properly
  - if this is not done and needs to be done then this would cause a significant problem in scoring
- allow for multiple models, occupancies, and insertions
- reevaluate treatment of water and hydrogens
