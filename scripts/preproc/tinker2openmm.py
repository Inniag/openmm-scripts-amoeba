#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Converts AMOEBA force field definitions from the Tinker to OpenMM format.

To use it, the user must provide both a Tinker PRM parameter file and a
Tinker XYZ coordinate file (note that the Tinker XYZ format is different from
the standard XYZ format). The XYZ file must contain coordinates and
connectivity information for the residue to be parameterised and the PRM file
must contain AMOEBA force field information for the same residue. Furthermore,
the XYZ file must specify the residue name in the first row (after the number
of atoms) and may not contain an additional comment row.

The user can specify the residue name from the command line. This will be
interpreted as the input file name without extension, so you need to make
sure that the files are names 'resname.xyz' and 'resname.prm'. Output will
be written to a PDB file and an XML wile of the same name. The PDB file
contains the converted atom coordinates and some default information in some
other fields. In addition it contains the CONECT records OpenMM requires for
HETATM coordinates (otherwise it will not know the bonds). The XML file will
contain the converted force field information including the residue
specification.

NOTE: By my understanding, OpenMM can combine multiple XML files, but
requires that the provider of the XML files makes sure that the atom types are
different between the different files. This is not very well documented, but I
think it is safest to make sure there are no internal mix-ups or ambiguities.
To this end, the user must specify a name prefix integer when calling this
script and he should make sure that this prefix prevents atom type name
overlaps with any of the other XML files he intends to use.

Note the following additional limitations:
- not all PRM file statements are implemented
  -- some are deliberately omitted and I checked that they are irrelevant for
     my own use cases, but the script will throw a warning to inform others
     users that they may want to worry about this
  -- the script will throw an error if unknown records are encountered, the
     user should definitely worry about these if they occur
- external bonds are not implemented, so the script can not convert polymer
  parameters
"""

import argparse
import shlex
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

# dictionary associating proton number with chemical element name:
element_symbols = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "O"
}


class TinkerXyzAtom:

    def __init__(self, atom_name, atom_type, index, x, y, z):
        """Constructor."""

        self.name = atom_name
        self.type = atom_type
        self.index = index
        self.x = x
        self.y = y
        self.z = z


class TinkerXyzBond:

    def __init__(self, bondfrom, bondto):
        """Constructor."""

        self.bondfrom = bondfrom
        self.bondto = bondto


class TinkerPrmAtom:

    def __init__(
            self,
            atom_type,
            atom_class,
            atom_symbol,
            description,
            atomic_number,
            atom_mass,
            num_bonds):
        """Constructor."""

        self.name = atom_type
        self.atom_class = atom_class
        self.symbol = atom_symbol
        self.description = description
        self.atomic_number = atomic_number
        self.mass = atom_mass
        self.num_bonds = num_bonds


class TinkerPrmVdw:

    def __init__(self, atom_class, sigma, epsilon, reduction):
        """Constructor."""

        self.atom_class = atom_class
        self.sigma = sigma
        self.epsilon = epsilon
        self.reduction = reduction


class TinkerPrmBond:

    def __init__(self, class_1, class_2, length, k):
        """Constructor."""

        self.class_1 = class_1
        self.class_2 = class_2
        self.length = length
        self.k = k


class TinkerPrmAngle:

    def __init__(self, class_1, class_2, class_3, k, angle_1):
        """Constructor."""

        self.class_1 = class_1
        self.class_2 = class_2
        self.class_3 = class_3
        self.k = k
        self.angle_1 = angle_1


class TinkerPrmStrbnd:

    def __init__(self, class_1, class_2, class_3, k_1, k_2):
        """Constructor."""

        self.class_1 = class_1
        self.class_2 = class_2
        self.class_3 = class_3
        self.k_1 = k_1
        self.k_2 = k_2


class TinkerPrmOpbend:

    def __init__(self, class_1, class_2, class_3, class_4, k):
        """Constructor."""

        self.class_1 = class_1
        self.class_2 = class_2
        self.class_3 = class_3
        self.class_4 = class_4
        self.k = k


class TinkerPrmTorsion:

    def __init__(
            self,
            class_1,
            class_2,
            class_3,
            class_4,
            amp_1,
            amp_2,
            amp_3,
            angle_1,
            angle_2,
            angle_3):
        """Constructor."""

        self.class_1 = class_1
        self.class_2 = class_2
        self.class_3 = class_3
        self.class_4 = class_4
        self.amp_1 = amp_1
        self.amp_2 = amp_2
        self.amp_3 = amp_3
        self.angle_1 = angle_1
        self.angle_2 = angle_2
        self.angle_3 = angle_3


class TinkerPrmMultipole:

    def __init__(
            self,
            atom_type,
            kz,
            kx,
            c_0,
            d_1,
            d_2,
            d_3,
            q_11,
            q_21,
            q_22,
            q_31,
            q_32,
            q_33):
        """Constructor."""

        self.atom_type = atom_type
        self.kz = kz
        self.kx = kx
        self.c_0 = c_0
        self.d_1 = d_1
        self.d_2 = d_2
        self.d_3 = d_3
        self.q_11 = q_11
        self.q_21 = q_21
        self.q_22 = q_22
        self.q_31 = q_31
        self.q_32 = q_32
        self.q_33 = q_33


class TinkerPrmPolarize:

    def __init__(
            self,
            atom_type,
            polarizability,
            thole,
            pgrp):
        """Constructor."""

        self.atom_type = atom_type
        self.polarizability = polarizability
        self.thole = thole
        self.pgrp = pgrp


class TinkerOpenmmConverter:

    xyz_atoms = list()
    xyz_bonds = list()

    prm_atom_records = list()
    prm_vdw_records = list()
    prm_bond_records = list()
    prm_angle_records = list()
    prm_strbnd_records = list()
    prm_opbend_records = list()
    prm_torsion_records = list()
    prm_multipole_records = list()
    prm_polarize_records = list()

    def write(self, xml_file, pdb_file, name_prefix):
        """Writes forcefield data to OpenMM format XML file."""

        # prefix type to avoid confusion when combining multiple XML files:
        # (note that for AMOEBA atom type must be an integer, so prefix in
        # leetspeak ;) )
        self.atom_type_prefix = str(name_prefix)
        self.atom_class_prefix = str(name_prefix)

        # create root XML element:
        self.ff = Element("ForceField")

        # add force field info:
        ff_info = SubElement(self.ff, "Info")
        ff_source = SubElement(ff_info, "Source")
        ff_source.text = self.prm_file

        # create OpenMM XML:
        self.xml_make_atom_types()
        self.xml_make_residue()
        self.xml_make_amoeba_vdw_force()
        self.xml_make_amoeba_bond_force()
        self.xml_make_amoeba_angle_force()
        self.xml_make_amoeba_strbnd_force()
        self.xml_make_amoeba_opbend_force()
        self.xml_make_amoeba_torsion_force()
        self.xml_make_amoeba_multipole_force()

        # create pretty XML:
        xmlstr = minidom.parseString(
            tostring(self.ff)
        ).toprettyxml(indent="   ")

        # write XML to file:
        with open(xml_file, 'w') as f:
            f.write(xmlstr)

        # write XYZ data to PDB file:
        with open(pdb_file, 'w') as f:

            f.write("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")  # noqa: E501

            # loop over atoms:
            for atm in self.xyz_atoms:
                f.write('{0: <6}'.format("HETATM"))           # record name
                f.write('{0: >5}'.format(atm.index))          # atom serial num
                f.write('{0: >1}'.format(""))
                f.write('{0: >4}'.format(atm.name))           # atom name
                f.write('{0: >1}'.format(" "))                # alt location
                f.write('{0: >3}'.format(self.resname[0:3]))  # residue name
                f.write('{0: >1}'.format(""))
                f.write('{0: >1}'.format("A"))                # chain
                f.write('{0: >4}'.format("1"))                # res seq number
                f.write('{0: >1}'.format(" "))                # insertion code
                f.write('{0: >3}'.format(""))
                f.write('{0: >8}'.format(str(atm.x)[0:5]))    # x
                f.write('{0: >8}'.format(str(atm.y)[0:5]))    # y
                f.write('{0: >8}'.format(str(atm.z)[0:5]))    # z
                f.write('{0: >6}'.format("0.00"))             # occupancy
                f.write('{0: >6}'.format("0.00"))             # temp factor
                f.write('{0: >10}'.format(""))
                f.write('{0: >2}'.format(""))                 # element
                f.write('{0: >2}'.format(""))                 # charge
                f.write("\n")

            # loop over bonds:
            for bnd in self.xyz_bonds:
                f.write('{0: <6}'.format("CONECT"))
                f.write('{0: >5}'.format(bnd.bondfrom))
                f.write('{0: >5}'.format(bnd.bondto))
                f.write("\n")

    def parse(self, xyz_file, prm_file):
        """Reads force field and residue data from Tinker XYZ and PRM files."""

        # add residues from Tinker XYZ file:
        self.parse_xyz(xyz_file)

        # add interaction info from Tinker PRM file:
        self.prm_file = prm_file
        self.parse_prm(prm_file)

    def parse_xyz(self, xyz_file):
        """Reads in and parses a Tinker XYZ file.

        Will read expected number of atoms from the first line and subsequently
        parse the remaining file line by line. For each line, appropriate atom
        and bond records are added to the internally maintained list.
        """

        # open Tinker XYZ file file:
        with open(xyz_file, "r") as f:

            # read first line from file:
            first_line = f.readline()

            # get number of atoms in this file:
            self.num_atoms = int(shlex.split(first_line)[0])

            # will assume that second element of first line is residue name:
            self.resname = shlex.split(first_line)[1]

            # go through file line by line:
            for line in f:

                # split line at white spaces:
                try:
                    split_line = shlex.split(line)
                except():
                    print("skipping line: " + line)

                # parse atom and connectivity record:
                self.parse_xyz_atom(split_line)

    def parse_prm(self, prm_file):
        """

        """

        # open Tinker PRM file:
        with open(prm_file, "r") as f:

            # go through file line by line:
            for line in f:

                # split line at white spaces:
                try:
                    split_line = shlex.split(line)
                except():
                    print("skipping line: " + line)
                    continue

                # skip empty lines:
                if len(split_line) == 0:
                    continue

                # inform user:
                if split_line[0] == "forcefield":
                    print(
                        "Reading Tinker PRM data fro forcefield: "
                        + split_line[1]
                    )

                # parse global van der Waals record types:
                elif split_line[0] == "vdwtype":
                    self.vdwtype = split_line[1]
                elif split_line[0] == "radiusrule":
                    self.radiusrule = split_line[1]
                elif split_line[0] == "radiustype":
                    self.radiustype = split_line[1]
                elif split_line[0] == "radiussize":
                    self.radiussize = split_line[1]
                elif split_line[0] == "epsilonrule":
                    self.epsilonrule = split_line[1]
                elif split_line[0] == "vdw-12-scale":
                    self.vdw_12_scale = split_line[1]
                elif split_line[0] == "vdw-13-scale":
                    self.vdw_13_scale = split_line[1]
                elif split_line[0] == "vdw-14-scale":
                    self.vdw_14_scale = split_line[1]
                elif split_line[0] == "vdw-15-scale":
                    self.vdw_15_scale = split_line[1]

                # parse global bond force records:
                elif split_line[0] == "bond-cubic":
                    self.bond_cubic = split_line[1]
                elif split_line[0] == "bond-quartic":
                    self.bond_quartic = split_line[1]

                # parse global angle force records:
                elif split_line[0] == "angle-cubic":
                    self.angle_cubic = split_line[1]
                elif split_line[0] == "angle-quartic":
                    self.angle_quartic = split_line[1]
                elif split_line[0] == "angle-pentic":
                    self.angle_pentic = split_line[1]
                elif split_line[0] == "angle-sextic":
                    self.angle_sextic = split_line[1]

                # parse out-of-plane-bending force parameters:
                elif split_line[0] == "opbendtype":
                    self.opbend_type = split_line[1]
                elif split_line[0] == "opbend-cubic":
                    self.opbend_cubic = split_line[1]
                elif split_line[0] == "opbend-quartic":
                    self.opbend_quartic = split_line[1]
                elif split_line[0] == "opbend-pentic":
                    self.opbend_pentic = split_line[1]
                elif split_line[0] == "opbend-sextic":
                    self.opbend_sextic = split_line[1]

                # parse global torsion force records:
                elif split_line[0] == "torsionunit":
                    self.torsion_unit = split_line[1]

                # parse global mutlipole parameters:
                elif split_line[0] == "direct-11-scale":
                    self.direct_11_scale = split_line[1]
                elif split_line[0] == "direct-12-scale":
                    self.direct_12_scale = split_line[1]
                elif split_line[0] == "direct-13-scale":
                    self.direct_13_scale = split_line[1]
                elif split_line[0] == "direct-14-scale":
                    self.direct_14_scale = split_line[1]
                elif split_line[0] == "mpole-12-scale":
                    self.mpole_12_scale = split_line[1]
                elif split_line[0] == "mpole-13-scale":
                    self.mpole_13_scale = split_line[1]
                elif split_line[0] == "mpole-14-scale":
                    self.mpole_14_scale = split_line[1]
                elif split_line[0] == "mpole-15-scale":
                    self.mpole_15_scale = split_line[1]
                elif split_line[0] == "mutual-11-scale":
                    self.mutual_11_scale = split_line[1]
                elif split_line[0] == "mutual-12-scale":
                    self.mutual_12_scale = split_line[1]
                elif split_line[0] == "mutual-13-scale":
                    self.mutual_13_scale = split_line[1]
                elif split_line[0] == "mutual-14-scale":
                    self.mutual_14_scale = split_line[1]
                elif split_line[0] == "polar-12-scale":
                    self.polar_12_scale = split_line[1]
                elif split_line[0] == "polar-13-scale":
                    self.polar_13_scale = split_line[1]
                elif split_line[0] == "polar-14-scale":
                    self.polar_14_scale = split_line[1]
                elif split_line[0] == "polar-15-scale":
                    self.polar_15_scale = split_line[1]
                elif split_line[0] == "polar-14-intra":
                    self.polar_14_intra = split_line[1]

                # parse complex record types:
                elif split_line[0] == "atom":
                    self.parse_prm_atom(split_line)
                elif split_line[0] == "vdw":
                    self.parse_prm_vdw(split_line)
                elif split_line[0] == "bond":
                    self.parse_prm_bond(split_line)
                elif split_line[0] == "angle":
                    self.parse_prm_angle(split_line)
                elif split_line[0] == "strbnd":
                    self.parse_prm_strbnd(split_line)
                elif split_line[0] == "opbend":
                    self.parse_prm_opbend(split_line)
                elif split_line[0] == "torsion":
                    self.parse_prm_torsion(split_line)
                elif split_line[0] == "multipole":
                    self.parse_prm_multipole(split_line, f)
                elif split_line[0] == "polarize":
                    self.parse_prm_polarize(split_line)

                # identify comments:
                elif split_line[0][0] == "#":
                    continue

                # warnings for not implemented records:
                elif split_line[0] == "dielectric":
                    print("WARNING: 'dielectric' records not implemented!")
                elif split_line[0] == "vdwpr":
                    print("WARNING: 'vdwpr' records not implemented!")
                elif split_line[0] == "pitors":
                    print("WARNING: 'pitors' records not implemented!")
                elif split_line[0] == "tortors":
                    print("WARNING: 'tortors' records not implemented!")
                elif split_line[0] == "polarization":
                    print("WARNING: 'polarization' records not implemented!")
                elif split_line[0] == "ureybrad":
                    print("WARNING: 'ureybrad' records not implemented!")

                # throw exception for unknown records:
                else:
                    raise Exception("Encountered unknown record type: " + line)

    def parse_xyz_atom(self, split_line):
        """ Parses one line from a Tinker XYZ file.

        This will add atom and bond records to the internal storage.
        """

        # add to internal list of atoms:
        self.xyz_atoms.append(TinkerXyzAtom(
            atom_name=split_line[1],
            atom_type=split_line[5],
            index=split_line[0],
            x=split_line[2],
            y=split_line[3],
            z=split_line[4]))

        # add to internal list of bonds:
        for idx in range(6, len(split_line), 1):
            self.xyz_bonds.append(TinkerXyzBond(split_line[0], split_line[idx]))

    def parse_prm_atom(self, split_line):
        """ Parses one atom record from a Tinker PRM file.

        This will add atom records to the internal storage.
        """

        self.prm_atom_records.append(TinkerPrmAtom(
            atom_type=split_line[1],
            atom_class=split_line[2],
            atom_symbol=split_line[3],
            description=split_line[4],
            atomic_number=split_line[5],
            atom_mass=split_line[6],
            num_bonds=split_line[7]))

    def parse_prm_vdw(self, split_line):
        """ Parses one vdw record from a Tinker PRM file. """

        # set reduction coefficient to 1 if not explicitly given:
        if len(split_line) == 5:
            reduction = split_line[4]
        else:
            reduction = 1.0

        # add vdW record to internal container:
        self.prm_vdw_records.append(TinkerPrmVdw(
            atom_class=split_line[1],
            sigma=split_line[2],
            epsilon=split_line[3],
            reduction=reduction))

    def parse_prm_bond(self, split_line):

        self.prm_bond_records.append(TinkerPrmBond(
            class_1=split_line[1],
            class_2=split_line[2],
            length=split_line[4],
            k=split_line[3]))

    def parse_prm_angle(self, split_line):

        self.prm_angle_records.append(TinkerPrmAngle(
            class_1=split_line[1],
            class_2=split_line[2],
            class_3=split_line[3],
            k=split_line[4],
            angle_1=split_line[5]))

    def parse_prm_strbnd(self, split_line):

        self.prm_strbnd_records.append(TinkerPrmStrbnd(
            class_1=split_line[1],
            class_2=split_line[2],
            class_3=split_line[3],
            k_1=split_line[4],
            k_2=split_line[5]))

    def parse_prm_opbend(self, split_line):

        self.prm_opbend_records.append(TinkerPrmOpbend(
            class_1=split_line[1],
            class_2=split_line[2],
            class_3=split_line[3],
            class_4=split_line[4],
            k=split_line[5]))

    def parse_prm_torsion(self, split_line):

        self.prm_torsion_records.append(TinkerPrmTorsion(
            class_1=split_line[1],
            class_2=split_line[2],
            class_3=split_line[3],
            class_4=split_line[4],
            amp_1=split_line[5],
            angle_1=split_line[6],
            amp_2=split_line[8],
            angle_2=split_line[9],
            amp_3=split_line[11],
            angle_3=split_line[12]))

    def parse_prm_multipole(self, split_line, f):

        # get first line parameters (atom types and charge):
        atom_type = split_line[1]
        kx = split_line[3]  # kz precedes kz in PRM file
        kz = split_line[2]
        c_0 = split_line[4]

        # get second line parameters (dipole moment):
        for line in f:
            split_line = shlex.split(line)
            d_1 = split_line[0]
            d_2 = split_line[1]
            d_3 = split_line[2]
            break

        # get third line parameters (quadrupole moments):
        for line in f:
            split_line = shlex.split(line)
            q_11 = split_line[0]
            break

        # get fourth line parameters (quadrupole moments):
        for line in f:
            split_line = shlex.split(line)
            q_21 = split_line[0]
            q_22 = split_line[1]
            break

        # get fifth line parameters (quadrupole moments):
        for line in f:
            split_line = shlex.split(line)
            q_31 = split_line[0]
            q_32 = split_line[1]
            q_33 = split_line[2]
            break

        # create multipole record:
        self.prm_multipole_records.append(TinkerPrmMultipole(
            atom_type=atom_type,
            kx=kx,
            kz=kz,
            c_0=c_0,
            d_1=d_1,
            d_2=d_2,
            d_3=d_3,
            q_11=q_11,
            q_21=q_21,
            q_22=q_22,
            q_31=q_31,
            q_32=q_32,
            q_33=q_33))

    def parse_prm_polarize(self, split_line):

        self.prm_polarize_records.append(TinkerPrmPolarize(
            atom_type=split_line[1],
            polarizability=split_line[2],
            thole=split_line[3],
            pgrp=split_line[4:]))

    def xml_make_residue(self):
        """ Creates an OpenMM XML residue from internal storage of bonds.

        This goes through the internally maintained lists of atoms and bonds
        and adds them to the XML residue element.
        """

        # create XML element for residue:
        # (note that it is assumed that we have one residue only)
        residues = SubElement(self.ff, "Residues")
        residue = SubElement(residues, "Residue")
        residue.set("name", self.resname)

        # add atoms to residue:
        for a in self.xyz_atoms:
            atom = SubElement(residue, "Atom")
            atom.set("name", a.name)
            atom.set("type", self.atom_type_prefix + str(a.type))

        # maintain dictionary of existing bonds to prevent reverse bonds:
        existing_bonds = dict()

        # add bonds to residue:
        # (note that OpenMM has zero-based index in residue entry)
        for b in self.xyz_bonds:

            if (
                b.bondfrom in existing_bonds and existing_bonds[b.bondfrom]
                == b.bondto
            ):
                continue

            if (
                b.bondto in existing_bonds and existing_bonds[b.bondto]
                == b.bondfrom
            ):
                continue

            # add to XML tree:
            bond = SubElement(residue, "Bond")
            bond.set("from", str(int(b.bondfrom) - 1))
            bond.set("to", str(int(b.bondto) - 1))

            # add to bond dictionary:
            existing_bonds[b.bondfrom] = b.bondto
            existing_bonds[b.bondto] = b.bondfrom

    def xml_make_atom_types(self):

        # create XML element for atom types:
        atom_types = SubElement(self.ff, "AtomTypes")

        # for each atom type in internal list create XML record:
        for a in self.prm_atom_records:
            atom_type = SubElement(atom_types, "Type")
            atom_type.set("name", self.atom_type_prefix + str(a.name))
            atom_type.set("class", self.atom_class_prefix + str(a.atom_class))
            atom_type.set("element", element_symbols[int(a.atomic_number)])
            atom_type.set("mass", a.mass)

    def xml_make_amoeba_vdw_force(self):

        # create XML element for vdw force and set properties:
        vdw_force = SubElement(self.ff, "AmoebaVdwForce")
        vdw_force.set("type", "BUFFERED-14-7")
        vdw_force.set("radiusrule", self.radiusrule)
        vdw_force.set("radiustype", self.radiustype)
        vdw_force.set("radiussize", self.radiussize)
        vdw_force.set("epsilonrule", self.epsilonrule)
        vdw_force.set("vdw-12-scale", self.vdw_12_scale)
        vdw_force.set("vdw-13-scale", self.vdw_13_scale)
        vdw_force.set("vdw-14-scale", self.vdw_14_scale)
        vdw_force.set("vdw-15-scale", self.vdw_15_scale)

        # conversion factors between Tinker and OpenMM:
        openmm_sigma_fac = 0.1      # Angstrom to nm
        openmm_epsilon_fac = 4.184  # kcal/mol to kJ/mol

        # for each vdw record create XML element:
        for elem in self.prm_vdw_records:
            vdw = SubElement(vdw_force, "Vdw")
            vdw.set("class", self.atom_class_prefix + str(elem.atom_class))
            vdw.set("epsilon", str(float(elem.epsilon) * openmm_epsilon_fac))
            vdw.set("sigma", str(float(elem.sigma) * openmm_sigma_fac))
            vdw.set("reduction", str(elem.reduction))

    def xml_make_amoeba_bond_force(self):

        # conversion factors between Tinker and OpenMM:
        openmm_cubic_fac = 10.0     # 1/Angstrom to 1/nm
        openmm_quartic_fac = 100.0  # 1/Angstrom^2 to 1/nm^2
        openmm_length_fac = 0.1     # Angstrom to nm
        openmm_k_fac = 418.4        # kcal/Angstrom^2 to kJ/nm^2

        # create XML element for bond force and set properties:
        bond_force = SubElement(self.ff, "AmoebaBondForce")
        bond_force.set(
            "bond-cubic", str(float(self.bond_cubic) * openmm_cubic_fac)
        )
        bond_force.set(
            "bond-quartic", str(float(self.bond_quartic) * openmm_quartic_fac)
        )

        # for each bond record create XML element:
        for elem in self.prm_bond_records:
            bond = SubElement(bond_force, "Bond")
            bond.set("class1", self.atom_class_prefix + str(elem.class_1))
            bond.set("class2", self.atom_class_prefix + str(elem.class_2))
            bond.set("length", str(float(elem.length) * openmm_length_fac))
            bond.set("k", str(float(elem.k) * openmm_k_fac))

    def xml_make_amoeba_angle_force(self):

        # conversion factors between Tinker and OpenMM:
        # NOTE: this is an odd one! OpenMM usually uses radians, but for the
        # AMOEBA plugin it uses degrees, see:
        # https://github.com/pandegroup/openmm/issues/1136
        # Tinker on the other hand uses radians by default and the values given
        # in the PRM files on the Tinker homepage definitely use radians in
        # agreement with the 2014 AMOEBA water model paper (as opposed to the
        # 2003 water model paper).
        openmm_k_fac = 0.001274519  # kcal/radian^2 to kJ/degree^2

        # create XML element for angle force and set properties:
        angle_force = SubElement(self.ff, "AmoebaAngleForce")
        angle_force.set("angle-cubic", self.angle_cubic)
        angle_force.set("angle-quartic", self.angle_quartic)
        angle_force.set("angle-pentic", self.angle_pentic)
        angle_force.set("angle-sextic", self.angle_sextic)

        # for each angle record create XML element:
        for elem in self.prm_angle_records:
            angle = SubElement(angle_force, "Angle")
            angle.set("class1", self.atom_class_prefix + str(elem.class_1))
            angle.set("class2", self.atom_class_prefix + str(elem.class_2))
            angle.set("class3", self.atom_class_prefix + str(elem.class_3))
            angle.set("k", str(float(elem.k) * openmm_k_fac))
            angle.set("angle1", elem.angle_1)

    def xml_make_amoeba_strbnd_force(self):

        # conversion factors between Tinker and OpenMM:
        # NOTE: this is an odd one! OpenMM usually uses radians, but for the
        # AMOEBA plugin it uses degrees, see:
        # https://github.com/pandegroup/openmm/issues/1136
        # Tinker on the other hand uses radians by default and the values given
        # in the PRM files on the Tinker homepage definitely use radians in
        # agreement with the 2014 AMOEBA water model paper (as opposed to the
        # 2003 water model paper).
        openmm_k_fac = 0.7302458    # kcal/radian^2 to kJ/degree^2

        # stretch bend units seems undefined, so define here:
        self.stretch_bend_unit = 1.0

        # create XML element for angle force and set properties:
        angle_force = SubElement(self.ff, "AmoebaStrbndForce")
        angle_force.set("stretchBendUnit", str(self.stretch_bend_unit))

        # for each angle record create XML element:
        for elem in self.prm_strbnd_records:
            angle = SubElement(angle_force, "StretchBend")
            angle.set("class1", self.atom_class_prefix + str(elem.class_1))
            angle.set("class2", self.atom_class_prefix + str(elem.class_2))
            angle.set("class3", self.atom_class_prefix + str(elem.class_3))
            angle.set("k1", str(float(elem.k_1) * openmm_k_fac))
            angle.set("k2", str(float(elem.k_2) * openmm_k_fac))

    def xml_make_amoeba_opbend_force(self):

        # conversion factors between Tinker and OpenMM:
        openmm_k_fac = 0.001274519  # kcal/degree^2 to kJ/radian^2 # fixme

        # stretch bend units seems undefined, so define here:
        self.stretch_bend_unit = 1.0

        # create XML element for angle force and set properties:
        angle_force = SubElement(self.ff, "AmoebaOutOfPlaneBendForce")
        angle_force.set("type", self.opbend_type)
        angle_force.set("opbend-cubic", self.opbend_cubic)
        angle_force.set("opbend-quartic", self.opbend_quartic)
        angle_force.set("opbend-pentic", self.opbend_pentic)
        angle_force.set("opbend-sextic", self.opbend_sextic)

        # for each angle record create XML element:
        for elem in self.prm_opbend_records:
            angle = SubElement(angle_force, "Angle")
            angle.set("class1", self.atom_class_prefix + str(elem.class_1))
            angle.set("class2", self.atom_class_prefix + str(elem.class_2))
            angle.set("class3", self.atom_class_prefix + str(elem.class_3))
            angle.set("class4", self.atom_class_prefix + str(elem.class_4))
            angle.set("k", str(float(elem.k) * openmm_k_fac))

    def xml_make_amoeba_torsion_force(self):

        # conversion factors between Tinker and OpenMM:
        openmm_amp_fac = 2.092          # kcal to kJ and a factor of 1/2
        openmm_angle_fac = 0.01745329   # degrees to radians

        # stretch bend units seems undefined, so define here:
        self.stretch_bend_unit = 1.0

        # create XML element for angle force and set properties:
        angle_force = SubElement(self.ff, "AmoebaTorsionForce")
        angle_force.set("torsionUnit", self.torsion_unit)

        # for each angle record create XML element:
        for elem in self.prm_torsion_records:
            angle = SubElement(angle_force, "Angle")
            angle.set("class1", self.atom_class_prefix + str(elem.class_1))
            angle.set("class2", self.atom_class_prefix + str(elem.class_2))
            angle.set("class3", self.atom_class_prefix + str(elem.class_3))
            angle.set("class4", self.atom_class_prefix + str(elem.class_4))
            angle.set("amp1", str(float(elem.amp_1) * openmm_amp_fac))
            angle.set("amp2", str(float(elem.amp_2) * openmm_amp_fac))
            angle.set("amp3", str(float(elem.amp_3) * openmm_amp_fac))
            angle.set("angle1", str(float(elem.angle_1) * openmm_angle_fac))
            angle.set("angle2", str(float(elem.angle_2) * openmm_angle_fac))
            angle.set("angle3", str(float(elem.angle_3) * openmm_angle_fac))

    def xml_make_amoeba_multipole_force(self):

        # conversion factors between Tinker and OpenMM:
        openmm_dipole_fac = 0.05291772          # bohr to nm
        openmm_quadrupole_fac = 0.0009334284    # TODO ?
        openmm_pol_fac = 0.001                  # Angstrom^3 to nm^3

        # stretch bend units seems undefined, so define here:
        self.stretch_bend_unit = 1.0

        # create XML element for mulitpole force and set properties:
        multipole_force = SubElement(self.ff, "AmoebaMultipoleForce")
        multipole_force.set("direct11Scale", self.direct_11_scale)
        multipole_force.set("direct12Scale", self.direct_12_scale)
        multipole_force.set("direct13Scale", self.direct_13_scale)
        multipole_force.set("direct14Scale", self.direct_14_scale)
        multipole_force.set("mpole12Scale", self.mpole_12_scale)
        multipole_force.set("mpole13Scale", self.mpole_13_scale)
        multipole_force.set("mpole14Scale", self.mpole_14_scale)
        multipole_force.set("mpole15Scale", self.mpole_15_scale)
        multipole_force.set("mutual11Scale", self.mutual_11_scale)
        multipole_force.set("mutual12Scale", self.mutual_12_scale)
        multipole_force.set("mutual13Scale", self.mutual_13_scale)
        multipole_force.set("mutual14Scale", self.mutual_14_scale)
        multipole_force.set("polar12Scale", self.polar_12_scale)
        multipole_force.set("polar13Scale", self.polar_13_scale)
        multipole_force.set("polar14Scale", self.polar_14_scale)
        multipole_force.set("polar15Scale", self.polar_15_scale)
        multipole_force.set("polar14Intra", self.polar_14_intra)

        # for each multipole record create XML element:
        for elem in self.prm_multipole_records:
            mp = SubElement(multipole_force, "Multipole")
            mp.set("type", self.atom_type_prefix + str(elem.atom_type))
            mp.set("kx", self.atom_type_prefix + str(elem.kx))
            mp.set("kz", self.atom_type_prefix + str(elem.kz))
            mp.set("c0", elem.c_0)
            mp.set("d1", str(float(elem.d_1) * openmm_dipole_fac))
            mp.set("d2", str(float(elem.d_2) * openmm_dipole_fac))
            mp.set("d3", str(float(elem.d_3) * openmm_dipole_fac))
            mp.set("q11", str(float(elem.q_11) * openmm_quadrupole_fac))
            mp.set("q21", str(float(elem.q_21) * openmm_quadrupole_fac))
            mp.set("q22", str(float(elem.q_22) * openmm_quadrupole_fac))
            mp.set("q31", str(float(elem.q_31) * openmm_quadrupole_fac))
            mp.set("q32", str(float(elem.q_32) * openmm_quadrupole_fac))
            mp.set("q33", str(float(elem.q_33) * openmm_quadrupole_fac))

        # for each polarize record create XML element:
        for elem in self.prm_polarize_records:
            pol = SubElement(multipole_force, "Polarize")
            pol.set("type", self.atom_type_prefix + str(elem.atom_type))
            pol.set(
                "polarizability",
                str(float(elem.polarizability) * openmm_pol_fac)
            )
            pol.set("thole", elem.thole)

            # can have varying number of polarisation groups:
            for i in range(0, len(elem.pgrp)):
                pol.set("pgrp" + str(i + 1), elem.pgrp[i])


def main():
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # parse Tinker files:
    tp = TinkerOpenmmConverter()
    tp.parse(str(args.resname) + ".xyz", str(args.resname) + ".prm")

    # write converted data to XML file:
    tp.write(
        str(args.resname) + ".xml", str(args.resname) + ".pdb",
        args.name_prefix
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-resname",
        nargs="?",
        const="residue",
        help="Base file name for both PRM and XYZ input.",
        default="residue"
    )
    parser.add_argument(
        "-name_prefix",
        type=int,
        required=True,
        help="Integer used to prefix atom types.",
        nargs="?"
    )
    args = parser.parse_args()
    argdict = vars(args)

    main(argdict)
