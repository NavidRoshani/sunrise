import sunrise as sun
import tequila as tq

xyzgeometry = """
    3
    Water
    O 0.000000 0.000000 0.000000
    H 0.757000 0.586000 0.000000
    H -0.757000 0.586000 0.000000
    """
graph = sun.Graph.parse_xyz(xyzgeometry)

edges = graph.get_spa_edges()
initial_guess = graph.get_orbital_coefficient_matrix()
geometry = graph.get_geometry_string()

mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', frozen_core=False).use_native_orbitals()
U = mol.make_ansatz(name="HCB-SPA", edges=edges)
opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=initial_guess.T, use_hcb=True,silent=True)

sun.plot_MO(opt.molecule)