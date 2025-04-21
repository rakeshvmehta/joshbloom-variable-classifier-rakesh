python3 ../Hierarchy/encode_hierarchy.py --h_file Test_files/test1_hierarchy.txt --m_file Test_files/test1_meta.txt --out Test_files/test1.parent-child.txt --out_names Test_files/test1_class_names.txt
python3 ../compute_class_embedding.py --hierarchy Test_files/test1.parent-child.txt --out Test_files/test1.unitsphere.pickle

python3 ../Hierarchy/encode_hierarchy.py --h_file Test_files/test2_hierarchy.txt --m_file Test_files/test2_meta.txt --out Test_files/test2.parent-child.txt --out_names Test_files/test2_class_names.txt
python3 ../compute_class_embedding.py --hierarchy Test_files/test2.parent-child.txt --out Test_files/test2.unitsphere.pickle

python3 ../Hierarchy/encode_hierarchy.py --h_file Test_files/test3_hierarchy.txt --m_file Test_files/test3_meta.txt --out Test_files/test3.parent-child.txt --out_names Test_files/test3_class_names.txt
python3 ../compute_class_embedding.py --hierarchy Test_files/test3.parent-child.txt --out Test_files/test3.unitsphere.pickle