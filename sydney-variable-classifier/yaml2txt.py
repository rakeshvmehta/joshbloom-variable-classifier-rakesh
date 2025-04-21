import yaml
import argparse

def readYaml(infile, outfile, level):
    predash = '-'*2*level + ' ' if level else ''
    with open(infile, 'r') as stream:
        try:
            top = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        outfile.write(predash+top['class']+'\n')
        writeSub(outfile, top, level+1)
    
def writeSub(outfile, top, level):
    predash = '-'*2*level + ' '
    for subclass in top['subclasses']:
        if 'ref' in subclass: 
            readYaml(subclass['ref'], outfile, level)
        else:
            outfile.write(predash+subclass['class']+'\n')
            if 'subclasses' in subclass: 
                writeSub(outfile, subclass, level+1)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert YAML file(s) to text file')

    parser.add_argument('--outfile', default='ConvertedYaml.txt',
                        help="Output text file name")
    parser.add_argument('--infile', default='top.yaml',
                        help="Intput yaml file name"
                        )

    args = parser.parse_args()

    with open(args.outfile,"w") as outfile:
        readYaml(args.infile, outfile, 0)