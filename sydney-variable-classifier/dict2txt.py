from tdtax import taxonomy as tax
import argparse
import json

def writeSub(outfile, tree, level):
    predash = '-'*2*level + ' '
    for subclass in tree['subclasses']:
        outfile.write(predash+subclass['class']+'\n')
        if 'subclasses' in subclass: 
            writeSub(outfile, subclass, level+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert JSON file(s) to text file')

    parser.add_argument('--outfile', default='ConvertedDict.txt',
                        help="Output text file name")
    parser.add_argument('--infile', default=None,
                        help="Intput json file name"
                        )

    args = parser.parse_args()
        
    if args.infile is not None: 
        infile = open(args.infile, 'r')
        tree = json.load(infile)
    else: 
        tree = tax
    
    with open(args.outfile,"w") as outfile:
        outfile.write(tree['class']+'\n')
        writeSub(outfile, tree, 1)
    
    if args.infile is not None: 
        infile.close()