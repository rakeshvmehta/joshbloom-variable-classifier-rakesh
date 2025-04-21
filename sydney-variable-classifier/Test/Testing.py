import subprocess
import sys

def testFunction(scriptargs, expected):
    p = subprocess.Popen(
        [sys.executable]+scriptargs,
        stdout=subprocess.PIPE
    )
    out = p.stdout.read()
    if scriptargs[0]=='../get_class_info.py':
        out=round(float(out.split()[-1]),2)
    correct = out==expected
    return correct, out

tests = [['../get_class_info.py', '--infile', 'Test_files/test1.unitsphere.pickle', '--classnames', 'Test_files/test1_class_names.txt','--class1', 'milo','--class2','dog'],
         ['../get_class_info.py', '--infile', 'Test_files/test1.unitsphere.pickle', '--classnames', 'Test_files/test1_class_names.txt','--class1', 'mouse','--class2','dog'],
         ['../get_class_info.py', '--infile', 'Test_files/test1.unitsphere.pickle', '--classnames', 'Test_files/test1_class_names.txt','--class1', 'dog','--class2','mouse'],
         ['../get_class_info.py', '--infile', 'Test_files/test1.unitsphere.pickle', '--classnames', 'Test_files/test1_class_names.txt','--class1', 'dog','--class2','dog'],
         ['../get_class_info.py', '--infile', 'Test_files/test1.unitsphere.pickle', '--classnames', 'Test_files/test1_class_names.txt','--class1', 'mouse','--class2','reptile'],
         ['../get_class_info.py', '--infile', 'Test_files/test1.unitsphere.pickle', '--classnames', 'Test_files/test1_class_names.txt','--class1', 'milo','--class2','animal'], #End of Test1
         ['../get_class_info.py', '--infile', 'Test_files/test2.unitsphere.pickle', '--classnames', 'Test_files/test2_class_names.txt','--class1', 'petal','--class2','leaf'],
         ['../get_class_info.py', '--infile', 'Test_files/test2.unitsphere.pickle', '--classnames', 'Test_files/test2_class_names.txt','--class1', 'petal','--class2','twig'],
         ['../get_class_info.py', '--infile', 'Test_files/test2.unitsphere.pickle', '--classnames', 'Test_files/test2_class_names.txt','--class1', 'leaf','--class2','bark'],
         ['../get_class_info.py', '--infile', 'Test_files/test2.unitsphere.pickle', '--classnames', 'Test_files/test2_class_names.txt','--class1', 'trunk','--class2','root'], #End of Test2
         ['../get_class_info.py', '--infile', 'Test_files/test3.unitsphere.pickle', '--classnames', 'Test_files/test3_class_names.txt','--class1', 'i','--class2','on'],
         ['../get_class_info.py', '--infile', 'Test_files/test3.unitsphere.pickle', '--classnames', 'Test_files/test3_class_names.txt','--class1', 'i','--class2','at'],
         ['../get_class_info.py', '--infile', 'Test_files/test3.unitsphere.pickle', '--classnames', 'Test_files/test3_class_names.txt','--class1', 'ions','--class2','abe'] #End of Test3
        ]

expected = [.75,.5,.5,1,.25,.25,.57,.67,.33,.5,.67,.12,.2]

if __name__ == '__main__':
    for i in range(len(tests)):
        correct, out = testFunction(tests[i],expected[i])
        if not correct:
            print("Test {} Failed, Outputted {}".format(i, out))
    print("End of tests")