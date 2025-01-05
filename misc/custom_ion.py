import arbor
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(0, 0, 0, 1), arbor.mpoint(1, 0, 0, 1), tag=1)
labels = arbor.label_dict()
decor = arbor.decor()
cell = arbor.cable_cell(tree, decor, labels)
m = arbor.single_cell_model(cell)
m.properties.set_ion(ion='x', valence=1, int_con=1, ext_con=0, rev_pot=0)
m.run(1)
