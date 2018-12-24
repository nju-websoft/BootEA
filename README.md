# BootEA
Source code and datasets for IJCAI-2018 paper "Bootstrapping Entity Alignment with Knowledge Graph Embedding".

## Dataset
Folder "dataset" contains the id files of DWY100K. 

The subfolder "mapping/0_3" contains the id files used in BootEA and MTransE while the subfolder "sharing/0_3" is for JAPE and IPTransE. They use 30% reference entity alignment as seeds. Id files in "sharing/0_3" is generated following the idea of parameter sharing that lets the two aligned enitites in seed alignment share the same id, while "mapping/0_3" does not.

The subfolder "mapping/0_3" inculdes the following files:
* ent_ids_1: entity ids in the source KG;
* ent_ids_2: entity ids in the target KG;
* ref_ent_ids: entity alignment for testing, list of pairs like (e_s, \t, e_t);
* sup_ent_ids: seed entity alignment (training data);
* rel_ids_1: relation ids in the source KG;
* rel_ids_2: relation ids in the target KG;
* triples_1: relation triples in the source KG;
* triples_2: relation triples in the target KG;

The subfolder "sharing/0_3" inculdes the following additional files:
* attr_ids_1: attribute ids in the source KG;
* attr_ids_2: attribute ids in the target KG;
* attr_range_type_1: attribute ranges in the source KG, list of pairs like (attribute id, \t, range code);
* attr_range_type_2: attribute ranges in the target KG;
* ent_attrs_1: entity attributes in the source KG; 
* ent_attrs_2: entity attributes in the target KG; 
* ref_ents: seed entity alignment denoted by URIs (training data);

DBP15K can be found [here](https://github.com/nju-websoft/JAPE). And, if you want the raw data of DWY100K, please email to zqsun.nju@gmail.com.

## Code
Folder "code" contains all codes of BootEA, in which:
* "AlignE.py" is the implementation of AlignE;
* "BootEA.py" is the implementation of BootEA;
* "param.py" is the config file.

### Dependencies
* Python 3
* Tensorflow 1.x 
* Scipy
* Numpy
* Graph-tool or igraph or NetworkX

> If you have any difficulty or question in running code and reproducing expriment results, please email to zqsun.nju@gmail.com and whu@nju.edu.cn.

## Citation
If you use this model or code, please cite it as follows:      
_Zequn Sun, Wei Hu, Qingheng Zhang, Yuzhong Qu. Bootstrapping Entity Alignment with Knowledge Graph Embedding. In: IJCAI 2018._



