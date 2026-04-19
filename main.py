from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange
from openff.units import unit

import argparse
import os
import time
from contextlib import contextmanager

# QUIET MODE (stderr suppressor)
@contextmanager
def suppress_stderr(enabled):
    if not enabled:
        yield
        return

    devnull_fd=os.open(os.devnull,os.O_WRONLY)
    old_stderr_fd=os.dup(2)
    os.dup2(devnull_fd,2)

    try:
        yield
    finally:
        os.dup2(old_stderr_fd,2)
        os.close(devnull_fd)

# CLI
def parse_arguments():
    parser=argparse.ArgumentParser(
        description="SDF -> GROMACS topology generator"
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--quiet",action="store_true")
    parser.add_argument("--no-progress",action="store_true")
    return parser.parse_args()

# LOAD MOLECULES
def load_molecules(input_file):
    print("Loading molecules from SDF...\n")
    molecules=Molecule.from_file(
        input_file,
        allow_undefined_stereo=True
    )
    print(f"Loaded {len(molecules)} molecules\n")
    return molecules

# UTILS
def ensure_conformer(mol):
    if len(mol.conformers)==0:
        mol.generate_conformers(n_conformers=1)

def safe_molecule_name(mol,index):
    return mol.name.strip() or f"mol_{index}"

# ETA
def compute_eta(start_total,processed,total):
    return (time.time()-start_total)/processed*(total-processed)

# PROCESS SINGLE MOLECULE
def process_molecule(
    mol,
    index,
    total,
    output_dir,
    forcefield,
    show_progress,
    start_total
):
    start_single=time.time()

    name=safe_molecule_name(mol,index)
    print(f"[{index}/{total}] Processing {name} ...")

    ensure_conformer(mol)
    mol.assign_partial_charges("am1bcc")

    interchange=Interchange.from_smirnoff(
        force_field=forcefield,
        topology=mol.to_topology()
    )

    interchange.box=[4,4,4]*unit.nanometer

    ligand_dir=os.path.join(output_dir,name)
    os.makedirs(ligand_dir,exist_ok=True)

    prefix=os.path.join(ligand_dir,name)
    interchange.to_gromacs(prefix)

    elapsed_single=time.time()-start_single

    if show_progress:
        eta=compute_eta(start_total,index,total)
        print(
            f"   done in {elapsed_single:.1f}s | "
            f"ETA ~ {eta/60:.1f} min remaining\n"
        )

# MAIN (ALL MOLECULES)
def main():
    args=parse_arguments()
    input_file,output_dir=args.input,args.output

    quiet_mode=args.quiet
    show_progress=not args.no_progress

    os.makedirs(output_dir,exist_ok=True)

    with suppress_stderr(quiet_mode):
        molecules=load_molecules(input_file)
        total=len(molecules)

        forcefield=ForceField(
            "openff_unconstrained-2.1.0.offxml"
        )

        start_total=time.time()

        for i,mol in enumerate(molecules,start=1):
            process_molecule(
                mol,
                i,
                total,
                output_dir,
                forcefield,
                show_progress,
                start_total
            )
    total_time=time.time()-start_total
    print(
        f"\nAll molecules processed successfully "
        f"in {total_time/60:.1f} minutes"
    )

# ENTRYPOINT
if __name__=="__main__":
    main()