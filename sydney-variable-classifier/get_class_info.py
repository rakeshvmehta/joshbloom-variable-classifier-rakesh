import pickle
import numpy as np
from fuzzywuzzy import fuzz, process
import argparse


def find_similarity(infile, classnames, class1, class2):
    with (open(infile, "rb")) as file:
        data = pickle.load(file)
        convert = {}
        with open(classnames) as conversion: 
            for conv in conversion: 
                num, classname = conv.strip().split()
                convert[classname] = int(num)
                
        classes = convert.keys()
        new_class1 = process.extractOne(class1, classes, scorer=fuzz.token_set_ratio)[0]
        new_class2 = process.extractOne(class2, classes, scorer=fuzz.token_set_ratio)[0]

        linear_labels = data['label2ind']
        embedding = data['embedding']
        
        #Compute similarity
        sim = np.matmul(embedding[linear_labels[convert[new_class1]]].T,embedding[linear_labels[convert[new_class2]]])
        print("Similarity between {} and {}: {}".format(new_class1, new_class2, sim))
    
def find_embedding(infile, classnames, class1):
    with (open(infile, "rb")) as file:
        data = pickle.load(file)
        convert = {}
        with open(classnames) as conversion: 
            for conv in conversion: 
                num, classname = conv.strip().split()
                convert[classname] = int(num)

        classes = convert.keys()
        new_class = process.extractOne(class1, classes, scorer=fuzz.token_set_ratio)[0]
                
        linear_labels = data['label2ind']
        embedding = data['embedding']
        
        #Find embedding
        emb = embedding[linear_labels[convert[new_class]]]
        print("{} Embedding: {}".format(new_class, emb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find embedding or similarity of class(es)')

    parser.add_argument('--infile', default="Embeddings/all_variable.unitsphere.pickle",
                        help="Input embeddings pickle file"
                        )
    parser.add_argument('--classnames', default="Hierarchy/class_names.txt",
                        help="Class names file produced by Encode_hierarchy.py"
                        )
    parser.add_argument('--class1', default=None,
                        help="Class")
    parser.add_argument('--class2', default=None,
                        help="If not given, find embedding of class1. Else, find similarity between class1 and class2")


    args = parser.parse_args()
        
    if args.class1 is not None and args.class2 is None: 
        find_embedding(args.infile, args.classnames, args.class1)
  
    elif args.class1 is not None: 
        find_similarity(args.infile, args.classnames, args.class1, args.class2)
        
    else: 
        print("Need input class")