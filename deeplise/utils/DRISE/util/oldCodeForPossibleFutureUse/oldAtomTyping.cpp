//NOTE THESE METHODS ARE NOT UPDATED TO SUPPORT NEW DATASTRUCTURES
//this simply types atoms based on atom types in a mol2 file (should be sybyl unless processed by fconv)
void classifyAtomsMOL2(std::vector<Particle*> &particles, std::string pathToFile, std::string fconvFlags){
  std::string fconv = "./bin/fconv";
  std::string mol2Path = pathToFile.substr(0, pathToFile.find(".") + 1) + "mol2";

  fconvConverter(pathToFile, mol2Path, fconv, fconvFlags);
  parseMOL2(mol2Path, particles);
}
//these methods call the above method and using fconv
void classifyAtomsFCONVISE(std::vector<Particle*> &particles, std::string pathToFile){
	std::cout<<"\n--------------------Atom Type----------------------" <<std::endl;
	time_t startTime = time(nullptr);
  std::string flags = " -W -f --m=0 ";

  classifyAtomsMOL2(particles, pathToFile, flags);


  std::map<std::string, std::string> atomTypeConv = {
		{"H.ac", "HYD"},   //acidic H (bonded to O.3ac, N.im, N.sam or N.ohac)
		{"H.onh", "HYD"},  //amide NH
		{"H.n", "HYD"},    //bonded to other nitrogens
		{"H.o", "HYD"},    //bonded to other oxygens
		{"H.0", "HYD"},    //all other hydrogens
		{"C.ar6p", "CAO"}, //sp2 carbon with a positive charged resonance structure in a protonated 6-membered heteroaromatic ring
		{"C.ar6x", "CAN"}, //sp2 carbon in a 6-membered heteroaromatic ring
		{"C.ar6", "CAN"},  //sp2 carbon in a benzene ring
		{"C.arp", "CAO"},  //sp2 carbon with a positive charged resonance structure in other protonated heteroaromatic rings
		{"C.arx", "CAN"},  //sp2 carbon in other heteroaromatics
		{"C.ar", "CAN"},   //sp2 carbon in other aromatics
		{"C.2r3o", "C2O"}, //carbonyl carbon in cyclopropanone or cyclopropenone
		{"C.2r3x", "C2O"}, //sp2 carbon in heterocyclic 3-membered rings
		{"C.2r3", "C2N"},  //sp2 carbon in 3-membered rings
		{"C.3r3x", "C3O"}, //sp3 carbon in heterocyclic 3-membered rings
		{"C.3r3", "C3N"},  //sp3 carbon in 3-membered rings
		{"C.1n", "C2O"},   //sp carbon in cyano groups
		{"C.1p", "C2O"},   //sp carbon with one heavy atom bonded
		{"C.1s", "C2O"},  //sp carbon with two heavy atoms bonded
		{"C.co2h", "C2O"}, //sp2 carbon in explicitly protonated COOH groups
		{"C.co2", "C2O"},  //sp2 carbon in COO-  groups (also set if protonation state is unknown)
		{"C.es", "C2O"},   //carbonyl carbon in ester groups or anhydrides
		{"C.hal", "C2O"},  //carbonyl carbon in acidhalogenides
		{"C.am", "C2O"},   //carbonyl carbon in amides
		{"C.o", "C2O"},    //other carbonyl carbon
		{"C.s", "C3O"},    //thionyl carbon
		{"C.gu", "CCO"},   //sp2 carbon in unprotonated guanidino groups
		{"C.guh", "CCO"},  //sp2 carbon in protonated guanidino groups (also set if protonation state is unknown)
		{"C.mi", "CCO"},   //sp2 carbon in unprotonated amidino groups
		{"C.mih", "CCO"},  //sp2 carbon in protonated amidino groups (also set if protonation state is unknown)
		{"C.n", "C2O"},    //sp2 carbon in imines
		{"C.2p", "C2N"},   //other sp2 carbon with one heavy atom bonded
		{"C.2s", "C2N"},   //other sp2 carbon with two heavy atoms bonded
		{"C.2t", "C2N"},   //other sp2 carbon with 3 heavy atoms bonded
		{"C.et", "C3O"},   //sp3 carbon in ethers
		{"C.ohp", "C3O"},  //sp3 carbon in primary alcoholes
		{"C.ohs", "C3O"},  //sp3 carbon in secondary alcoholes
		{"C.oht", "C3O"},  //sp3 carbon in tertiary alcoholes
		{"C.3n", "C3O"},   //other sp3 carbon bonded to nitrogen
		{"C.3p", "C3N"},   //other sp3 carbon with one heavy atom bonded
		{"C.3s", "C3N"},   //other sp3 carbon with two heavy atoms bonded
		{"C.3t", "C3N"},   //other sp3 carbon with 3 heavy atoms bonded
		{"C.3q", "C3N"},   //other sp3 carbon with 4 heavy atoms bonded
		{"N.ar6p", "NAC"}, //positive charged nitrogen in 6-membered aromatics (e.g. pyridinium or NAD+)
		{"N.ar6", "NAV"},  //NOTE sp2 nitrogen in 6-membered aromatics
		{"N.arp", "NAV"},  //sp2 nitrogen in protonated aromatics (e.g both nitrogens in protonated imidazole
		{"N.ar2", "NAV"},  //NOTE****sp2 nitrogen in aromatics with two bonded atoms (corresponding to sybyl type N.2)
		{"N.ar3", "NAV"},  //NOTE sp2 nitrogen in aromatics with 3 heavy atoms (corresponding to sybyl type N.pl3)
		{"N.ar3h", "NAV"}, //NOTE H- sp2 nitrogen in aromatics with 2 heavy atoms and one hydrogen (corresponding to sybyl type N.pl3)
		{"N.r3", "N3E"},   //sp3 in aziridine or azirene rings
		{"N.az", "N2E"},   //middle nitrogen in azides
		{"N.1", "N2E"},    //other sp nitrogen
		{"N.o2", "NCC"},   //in nitro groups
		{"N.ohac", "NCE"}, //in hydroxamic acids
		{"N.oh", "N3O"},   //,E->Oin hydroxylamines
		{"N.ims", "NCO"},  //,E->OH-imide nitrogen with two heavy atoms bonded
		{"N.imt", "NCO"},  //,E->O****imide nitrogen with 3 heavy atoms bonded
		{"N.amp", "NCO"},  //H,E->O****carbon- or thionamide with one heavy atom bonded
		{"N.ams", "NCO"},  //H,E->O****carbon- or thionamide with two heavy atoms bonded
		{"N.amt", "NCO"},  //,E->Ocarbon- or thionamide with 3 heavy atoms bonded
		{"N.samp", "NCO"}, //,E->OH-sulfonamide with one heavy atom bonded
		{"N.sams", "NCO"}, //,E->OH- sulfonamide with two heavy atoms bonded
		{"N.samt", "NCO"}, //,E->Osulfonamide with 3 heavy atoms bonded
		{"N.gu1", "NCE"},  //H- NH in unprotonated guanidino group (only if explicitly protonated)
		{"N.gu2", "NCE"},  //H- NH2 in unprotonated guanidino group (only if explicitly protonated)
		{"N.guh", "NCC"},  //nitrogen in protonated guanidino group (also set if protonation state is unknown)
		{"N.mi1", "NCE"},  //H- NH in unprotonated amidino group (only if explicitly protonated)
		{"N.mi2", "NCE"},  //H- NH2 in unprotonated amidino group (only if explicitly protonated)
		{"N.mih", "NCC"}, //nitrogen in protonated amidino group (also set if protonation state is unknown)
		{"N.aap", "NLE"},  //H- primary aromatic amine (hybridization can't be determined exactly)
		{"N.aas2", "NLE"}, //H- sp2 hybridized secondary aromatic amine
		{"N.aas3", "NLE"}, //H- sp3 hybridized secondary aromatic amine
		{"N.aat2", "NLE"}, //sp2 hybridized tertiary aromatic amine
		{"N.aat3", "NLE"}, //sp3 hybridized tertiary aromatic amine
		{"N.2n", "NCE"},   //sp2 nitrogen bonded to another nitrogen
		{"N.2p", "N2E"},   //h- other sp2 nitrogen with one heavy atom
		{"N.2s", "N2E"},   //other sp2 nitrogen with two heavy atoms
		{"N.2t", "N2C"},   //other sp2 nitrogen with three heavy atoms
		{"N.3n", "N3E"},   //sp3 nitrogen bonded to another nitrogen
		{"N.3p", "N3C"},   //sp3 nitrogen with one heavy atom bonded
		{"N.3s", "N3E"},   //H- sp3 nitrogen with two heavy atoms bonded
		{"N.3t", "N3E"},   //sp3 nitrogen with 3 heavy atoms bonded
		{"N.4q", "N3C"},   //sp3 nitrogen with 4 bonded heavy atoms
		{"N.4h", "N3C"},   //sp3 nitrogen with 4 bonded atoms (at least 1 hydrogen)
		{"O.ar", "OAE"},   //aromatic oxygen
		{"O.r3", "O3E"},   //in oxiran ring
		{"O.h2o", "O3E"},  //H- water oxygen
		{"O.n", "OCC"},    //oxygen in nitro groups
		{"O.noh", "O3E"},  //H- sp3 oxygen in hydroxylamine or hydroxamic acid
		{"O.2co2", "OCE"}, //sp2 oxygen in COOH (sp2 bonded to C.co2h)
		{"O.2es", "OCE"},  //sp2 oxygen in esters or anhydrids
		{"O.2hal", "OCE"}, //sp2 oxygen in acidhalogenides
		{"O.am", "OCE"},   //in carbonamides
		{"O.co2", "OCC"},  //in COO-  or CSO-
		{"O.2po", "O2E"},  //sp2 oxygen in P=O (non deprotonated groups)
		{"O.2so", "O2E"},  //sp2 oxygen in S=O (non deprotonated groups)
		{"O.2p", "OCC"},   //sp2 oxygen in OPO3H- or PO3H- or POO-
		{"O.2s", "OCC"},   //sp2 oxygen in OSO3- or SO3- or POO- or deprotonated sulfonamides
		{"O.3po", "O2E"},  //sp3 oxygen with 2 heavy atoms bonded to at least one phosphor
		{"O.3so", "O2E"},  //sp3 oxygen with 2 heavy atoms bonded to at least one sulfur
		{"O.carb", "O2E"}, //in other carbonyl groups
		{"O.o", "O3E"},    //in peroxo groups
		{"O.3ac", "OCE"},  //H- OH in COOH, CSOH, POOHOH, POOH or SOOOH
		{"O.ph", "OLE"},   //H- phenolic hydroxyl group
		{"O.3oh", "O3E"},  //H- hydroxyl group
		{"O.3es", "OCE"},  //sp3 oxygen in esters or anhydrids
		{"O.3eta", "OLE"}, //****aromatic ether
		{"O.3et", "O3E"},  //aliphatic ether
		{"S.ar", "SAE"},   //aromatic sulfur
		{"S.r3", "S3E"},   //in thiiran ring
		{"S.thi", "S2E"},  //thionyl group
		{"S.o", "S2E"},    //in SO
		{"S.o2h", "SCE"},  //in protonated sulfonamide or other SO2
		{"S.o3h", "SCE"},  //in SO3
		{"S.o4h", "SCE"},  //in OSO3
		{"S.o2", "SCE"},   //in SO2 or deprotonated sulfonamides (or unknown protonation state)
		{"S.o3", "SCC"},   //in SO3- (or unknown protonation state)
		{"S.o4", "SCC"},   //in OSO3- (or unknown protonation state)
		{"S.2", "SCC"},    //in CSO-  COS-  or other sp2
		{"S.sh", "S3N"},   //H- in SH groups
		{"S.s", "S3E"},    //H- in S-S bonds
		{"S.3", "S3N"},    //other sp3 sulfur
		{"P.r3", "PHO"},  //in phosphiran rings
		{"P.o", "PHO"},   //in PO
		{"P.o2h", "PHO"}, //in not deprotonated PO2 groups
		{"P.o3h", "PHO"}, //in not deprotonated PO3 groups
		{"P.o4h", "PHO"}, //in not deprotonated PO4 groups
		{"P.o2", "PHO"},  //in deprotonated PO2 groups (or unknown protonation state)
		{"P.o3", "PHO"},  //in deprotonated PO3 groups (or unknown protonation state)
		{"P.o4", "PHO"},  //in deprotonated PO4 groups (or unknown protonation state)
		{"P.3", "PHO"},   //other sp3
		{"F.0", "HAL"},   //bonded fluor
		{"F.i", "HAL"},   //fluor ion
		{"Cl.0", "HAL"},  //bonded chlorine
		{"Cl.i", "HAL"},  //chlorine ion
		{"Br.0", "HAL"},  //bonded bromine
		{"Br.i", "HAL"},  //bromine ion
		{"I.0", "HAL"},   //bonded iod
		{"I.i", "HAL"},   //iod ion
		{"Li", "MET"},
		{"Na", "MET"},
		{"Mg", "MET"},
		{"Al", "MET"},
		{"Si", "MET"},
		{"K", "MET"},
		{"Ca", "MET"},
		{"Cr.th", "MET"},
		{"Cr.oh", "MET"},
		{"Mn", "MET"},
		{"Fe", "MET"},
		{"Co", "MET"},
		{"Cu", "MET"},
		{"Zn", "MET"},
		{"Se", "SMT"},
		{"Mo", "MET"},
		{"Sn", "MET"},
		{"Ni", "MET"},
		{"Hg", "MET"},
		{"B", "SMT"},
		{"As", "SMT"}
  };

	int counter = 0;
	//leading and trailing spaces are necessary for .find in next loop for met and semimet

	std::string pp = " ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR ";
	std::string modified_pp = " NCY N2C MVA DSN ";

	std::string element;
	std::string resChecker;
  /* Assign the atom types */
	std::string metalChecker = "";
	bool error = false;
  std::string sybylAtomType;
	for (auto atom = particles.begin(); atom != particles.end(); ++atom) {
		element = (*atom)->element;
		metalChecker = " " + element + " ";
		resChecker = " " + (*atom)->resName + " ";
		if(pp.find(resChecker) || modified_pp.find(resChecker)){
			if((*atom)->atomName == "C"){
				(*atom)->atomType = "CCO";
				continue;
			}
			else if((*atom)->atomName == "N"){
				(*atom)->atomType = "NCO";
				continue;
			}
			else if((*atom)->atomName == "O"){
				(*atom)->atomType = "OCE";
			}
			else if((*atom)->atomName == "OXT"){
				(*atom)->atomType = "OCC";
				auto temp = atom;
				while(temp != particles.begin() && (*temp)->atomName != "O") temp--;
				(*temp)->atomType = "OCC";
				continue;
			}
		}
		if(((*atom)->atomType).length() > 0 && atomTypeConv[((*atom)->atomType)].length() > 0) {
      sybylAtomType = (*atom)->atomType;
			(*atom)->atomType = atomTypeConv[((*atom)->atomType)];

	    // Neutral Carbon bonded to NOPS
	    if ( ((*atom)->element == "C") && ((*atom)->atomType.at(2) == 'N')) {
	      for (auto boundAtom = (*atom)->boundAtoms.begin(); boundAtom != (*atom)->boundAtoms.end(); ++boundAtom) {
					if ((*boundAtom)->element == "N" || (*boundAtom)->element == "O" || (*boundAtom)->element == "P") {
						(*atom)->atomType.replace(2,1,"O");
	        }
	      }
	    }
	    // sp3 CNOS adjacent to an sp2 or aromatic
	    if ( (*atom)->atomType.at(1) == '3')   {
	      for (auto boundAtom = (*atom)->boundAtoms.begin(); boundAtom != (*atom)->boundAtoms.end(); ++boundAtom) {
					if ((*boundAtom)->atomType.find("ar") != std::string::npos || (*boundAtom)->atomType.at(1) == 'A') {
						(*atom)->atomType.replace(1,1,"L");
						break;
					}
	        else if ("N.amt" == (*boundAtom)->atomType || "NCO" == (*boundAtom)->atomType || "C.3s" == (*boundAtom)->atomType ||  "C3N" == (*boundAtom)->atomType ) {
						/* This change fixes the delta carbon mistake on Proline */
						counter++;
	        }
	        if (counter == 2 && ((*atom)->boundAtoms).size() == 2) {
						(*atom)->atomType.replace(1,1,"3");
						break;
	        }
	      }
				counter = 0;
	   	}

	    // sp2 CNOS adjacent to an sp2 or aromatic
	    if((*atom)->atomType.at(1) == '2' && sybylAtomType != "C.co2" )  {
	      for (auto boundAtom = (*atom)->boundAtoms.begin(); boundAtom != (*atom)->boundAtoms.end(); ++boundAtom) {
          if ((*boundAtom)->atomType.find("ar") != std::string::npos || (*boundAtom)->atomType.at(1) == 'A') {
						(*atom)->atomType.replace(1,1,"C");
						break;
					}
	      }
			}
			//this is to fix incorrect NAE classification in HIS
			if((*atom)->resName == "HIS" && ((*atom)->atomName == "ND1" || (*atom)->atomName == "NE2")){
				(*atom)->atomType = "NAV";
			}
		}
		else{
			if(element == "H"){
				(*atom)->atomType = "HYD";
			}
			else if(element == "F" || element == "CL" || element == "BR" || element == "I" || element == "AT"){
				(*atom)->atomType = "HAL";
			}
			else if(element == "P") (*atom)->atomType = "PHO";
			else if(element == "HE" || element == "NE" || element == "AR" || element == "KR" || element == "XE" || element == "RN"){
				(*atom)->atomType = "NOB";
			}
			else if(metals.find(metalChecker) != std::string::npos){
				(*atom)->atomType = "MET";
			}
			else if(semimetals.find(metalChecker) != std::string::npos){
				(*atom)->atomType = "SMT";
			}
			else if((*atom)->element == "X" || (*atom)->resName == "UNK"){
				(*atom)->atomType = "UNK";
			}
		}
		if(((*atom)->atomType == "MET" || (*atom)->atomType == "SMT") && (*atom)->charge != 0){
			if((*atom)->charge < 0){
				(*atom)->atomType = "M" + std::to_string((*atom)->charge);
			}
			else{
				(*atom)->atomType = "M+" + std::to_string((*atom)->charge);
			}
		}
	}
	std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for atomType classification.\n\n";
}
void classifyAtomsFCONV(std::vector<Particle*> &particles, std::string pathToFile){
  std::cout<<"\n--------------------Atom Type----------------------" <<std::endl;
  time_t startTime = time(nullptr);
  std::string flags = " -W -f --m=0 ";

  classifyAtomsMOL2(particles, pathToFile, flags);

  std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for atomType classification.\n\n";

}
void classifyAtomsSybyl(std::vector<Particle*> &particles, std::string pathToFile){
  std::cout<<"\n--------------------Atom Type----------------------" <<std::endl;
	time_t startTime = time(nullptr);
  std::string flags = " -W -f --m=1 ";

  classifyAtomsMOL2(particles, pathToFile, flags);

	std::cout << "time elapsed = " << difftime(time(nullptr), startTime) <<" seconds for atomType classification.\n\n";

}
